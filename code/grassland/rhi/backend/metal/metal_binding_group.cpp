#include "grassland/rhi/backend/metal/metal_binding_group.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_sampler.h"
#include "grassland/rhi/backend/backend_shader_cache.h"
#include "grassland/rhi/backend/metal/metal_acceleration_structure.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_image.h"
#include "grassland/rhi/backend/metal/metal_sampler.h"
#include "grassland/rhi/backend/metal/metal_swap_chain_image.h"
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_buffer.h"
#include "grassland/rhi/rhi_device.h"
#include "grassland/rhi/rhi_image.h"

namespace grassland::rhi::backend::metal {
uint32_t MetalArgumentCount(const BindingLayoutDesc &layout) {
  uint64_t count = 0;
  for (auto b : layout.bindings)
    count += b.count;
  // Bound allocation and index arithmetic before constructing native descriptors.
  return count < UINT32_MAX ? uint32_t(count) : 0;
}

bool ValidateMetalBindingLayout(MTL::Device *device, const BindingLayoutDesc &layout) {
  for (auto binding : layout.bindings)
    if (binding.type == BINDING_ACCELERATION_STRUCTURE && !device->supportsRaytracing())
      return false;
  auto count = MetalArgumentCount(layout);
  if (!ValidateBindingLayout(layout) || (!layout.bindings.empty() && !count) ||
      (uint64_t(count) + 1) * sizeof(uint64_t) > device->maxBufferLength())
    return false;
  uint64_t samplers = 0, buffers = 1, textures = 0;
  for (auto b : layout.bindings) {
    if (b.type == BINDING_SAMPLER)
      samplers += b.count;
    else if (b.type >= BINDING_TEXTURE2D)
      textures += b.count;
    else
      buffers += b.count;
  }
  // Tier 1 includes the iOS simulator. Account for the generated size buffer.
  if (device->argumentBuffersSupport() == MTL::ArgumentBuffersTier1 &&
      (buffers > 31 || textures > 31 || samplers > 16)) {
    ReportError(-1, "Binding layout exceeds Metal Tier 1 argument-buffer limits");
    return false;
  }
  return samplers <= device->maxArgumentBufferSamplerCount();
}

MetalBindingVersionImpl::MetalBindingVersionImpl(IMetalDevice *device, const BindingLayoutDesc &layout)
    : device_(device) {
  InitializeBindingVersion(layout);
}

MetalBindingVersionImpl *MetalBindingVersionImpl::Create(IMetalDevice *device, const BindingLayoutDesc &layout) {
  if (!ValidateMetalBindingLayout(device->NativeDevice(), layout))
    return nullptr;

  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto result = std::unique_ptr<MetalBindingVersionImpl>(new MetalBindingVersionImpl(device, layout));
  std::vector<MTL::ArgumentDescriptor *> descriptors;
  uint32_t offset = 0;
  for (auto b : result->layout_.bindings) {
    auto d = MTL::ArgumentDescriptor::argumentDescriptor();
    d->setIndex(offset);
    d->setArrayLength(b.count);
    d->setDataType(b.type == BINDING_ACCELERATION_STRUCTURE ? MTL::DataTypeInstanceAccelerationStructure
                   : b.type == BINDING_SAMPLER              ? MTL::DataTypeSampler
                   : b.type >= BINDING_TEXTURE2D            ? MTL::DataTypeTexture
                                                            : MTL::DataTypePointer);
    if (b.type == BINDING_TEXTURE2D || b.type == BINDING_RW_TEXTURE2D)
      d->setTextureType(MTL::TextureType2D);
    d->setAccess(b.type == BINDING_RW_STRUCTURED_BUFFER || b.type == BINDING_RW_BYTE_ADDRESS_BUFFER ||
                         b.type == BINDING_RW_TEXTURE2D
                     ? MTL::ArgumentAccessReadWrite
                     : MTL::ArgumentAccessReadOnly);
    descriptors.push_back(d);
    result->offsets_.push_back(offset);
    result->initialized_.emplace_back(b.count, false);
    offset += b.count;
  }
  // SPIRV-Cross uses one size array per descriptor set for GetDimensions.
  auto sizes = MTL::ArgumentDescriptor::argumentDescriptor();
  sizes->setIndex(offset);
  sizes->setDataType(MTL::DataTypePointer);
  sizes->setAccess(MTL::ArgumentAccessReadOnly);
  descriptors.push_back(sizes);
  result->encoder_ = NS::TransferPtr(device->NativeDevice()->newArgumentEncoder(
      NS::Array::array(reinterpret_cast<const NS::Object *const *>(descriptors.data()), descriptors.size())));
  if (!result->encoder_)
    return nullptr;
  result->arguments_ = NS::TransferPtr(
      device->NativeDevice()->newBuffer(result->encoder_->encodedLength(), MTL::ResourceStorageModeShared));
  result->sizes_ = NS::TransferPtr(
      device->NativeDevice()->newBuffer(std::max(1u, offset) * sizeof(uint32_t), MTL::ResourceStorageModeShared));
  if (!result->arguments_ || !result->sizes_)
    return nullptr;
  std::memset(result->arguments_->contents(), 0, result->arguments_->length());
  std::memset(result->sizes_->contents(), 0, result->sizes_->length());
  result->encoder_->setArgumentBuffer(result->arguments_.get(), 0);
  result->encoder_->setBuffer(result->sizes_.get(), 0, offset);
  return result.release();
}

ObjectPtr<IMetalBindingVersion> MetalBindingVersionImpl::Clone() const {
  auto copy = ObjectPtr<MetalBindingVersionImpl>::Adopt(Create(device_, layout_));
  if (!copy)
    return {};
  copy->references_ = references_;
  copy->image_views_ = image_views_;
  copy->native_views_ = native_views_;
  copy->initialized_ = initialized_;
  std::memcpy(copy->arguments_->contents(), arguments_->contents(), arguments_->length());
  std::memcpy(copy->sizes_->contents(), sizes_->contents(), sizes_->length());
  copy->encoder_->setBuffer(copy->sizes_.get(), 0, MetalArgumentCount(layout_));
  return ObjectPtr<IMetalBindingVersion>::Retain(copy.Get());
}

bool MetalBindingVersionImpl::IsInitialized(size_t b, uint32_t e) const {
  return initialized_[b][e];
}

bool MetalBindingVersionImpl::WriteDescriptors(size_t b, uint32_t first, const std::vector<BufferView> &values) {
  for (auto v : values)
    if (v.offset % (layout_.bindings[b].type == BINDING_UNIFORM_BUFFER ? 256 : 4) || v.size > UINT32_MAX)
      return false;
  for (size_t i = 0; i < values.size(); ++i) {
    auto v = values[i];
    uint32_t index = offsets_[b] + first + i;
    encoder_->setBuffer(rhi::QueryInterfacePtr<IMetalBuffer>(v.buffer).Get()->NativeBuffer(), v.offset, index);
    static_cast<uint32_t *>(sizes_->contents())[index] = uint32_t(v.size);
    initialized_[b][first + i] = true;
  }
  return true;
}

bool MetalBindingVersionImpl::WriteDescriptors(size_t b, uint32_t first, const std::vector<ImageView> &values) {
  std::vector<NS::SharedPtr<MTL::Texture>> views;
  for (const auto &view : values) {
    auto image = rhi::QueryInterfacePtr<IMetalImage>(view.image)->NativeImage();
    auto texture =
        NS::TransferPtr(image->newTextureView(image->pixelFormat(), MTL::TextureType2D,
                                              NS::Range::Make(view.base_mip, view.mip_count), NS::Range::Make(0, 1)));
    if (!texture)
      return false;
    views.push_back(std::move(texture));
  }
  for (size_t i = 0; i < values.size(); ++i) {
    encoder_->setTexture(views[i].get(), offsets_[b] + first + i);
    native_views_[b][first + i] = std::move(views[i]);
    initialized_[b][first + i] = true;
  }
  return true;
}

bool MetalBindingVersionImpl::WriteDescriptors(size_t b, uint32_t first, const std::vector<ISampler *> &values) {
  for (size_t i = 0; i < values.size(); ++i) {
    encoder_->setSamplerState(rhi::QueryInterfacePtr<IMetalSampler>(values[i]).Get()->NativeSampler(),
                              offsets_[b] + first + i);
    initialized_[b][first + i] = true;
  }
  return true;
}

bool MetalBindingVersionImpl::WriteDescriptors(size_t b,
                                               uint32_t first,
                                               const std::vector<ITopLevelAccelerationStructure *> &values) {
  std::vector<MTL::AccelerationStructure *> handles;
  handles.reserve(values.size());
  for (auto value : values) {
    auto native = rhi::QueryInterfacePtr<IMetalTopLevelAccelerationStructure>(value).Get();
    if (!native || native->Owner() != device_)
      return false;
    handles.push_back(native->NativeStructure());
  }
  for (size_t i = 0; i < values.size(); ++i) {
    encoder_->setAccelerationStructure(handles[i], offsets_[b] + first + i);
    initialized_[b][first + i] = true;
  }
  return true;
}

void MetalBindingVersionImpl::Bind(MTL::RenderCommandEncoder *encoder, uint32_t group) const {
  encoder->setVertexBuffer(arguments_.get(), 0, group);
  encoder->setFragmentBuffer(arguments_.get(), 0, group);
  encoder->useResource(sizes_.get(), MTL::ResourceUsageRead, MTL::RenderStageVertex | MTL::RenderStageFragment);
  for (size_t b = 0; b < layout_.bindings.size(); ++b) {
    auto desc = layout_.bindings[b];
    uint32_t writable_texture = offsets_[b];
    if (desc.type == BINDING_RW_TEXTURE2D) {
      const bool tier1 = GetShaderCacheSettings().ios ||
                         device_->NativeDevice()->argumentBuffersSupport() == MTL::ArgumentBuffersTier1;
      for (const auto &view : native_views_[b]) {
        if (tier1) {
          auto texture = view.get();
          encoder->setVertexTexture(texture, writable_texture);
          encoder->setFragmentTexture(texture, writable_texture);
        }
        ++writable_texture;
      }
    }
    if (desc.type == BINDING_SAMPLER)
      continue;  // Samplers are retained objects, not MTL::Resource residency entries.
    MTL::ResourceUsage usage = MTL::ResourceUsageRead;
    if (desc.type == BINDING_RW_STRUCTURED_BUFFER || desc.type == BINDING_RW_BYTE_ADDRESS_BUFFER ||
        desc.type == BINDING_RW_TEXTURE2D)
      usage |= MTL::ResourceUsageWrite;
    MTL::RenderStages stages = 0;
    if (desc.visibility & SHADER_VISIBILITY_VERTEX)
      stages |= MTL::RenderStageVertex;
    if (desc.visibility & SHADER_VISIBILITY_FRAGMENT)
      stages |= MTL::RenderStageFragment;
    if (!stages)
      continue;
    for (auto &resource : references_[b]) {
      MTL::Resource *native =
          desc.type == BINDING_ACCELERATION_STRUCTURE
              ? static_cast<MTL::Resource *>(rhi::QueryInterfacePtr<IMetalTopLevelAccelerationStructure>(resource.Get())
                                                 .Get()
                                                 ->NativeStructure())
          : desc.type <= BINDING_RW_BYTE_ADDRESS_BUFFER
              ? static_cast<MTL::Resource *>(rhi::QueryInterfacePtr<IMetalBuffer>(resource.Get()).Get()->NativeBuffer())
              : static_cast<MTL::Resource *>(rhi::QueryInterfacePtr<IMetalImage>(resource.Get()).Get()->NativeImage());
      encoder->useResource(native, usage, stages);
    }
    for (const auto &view : native_views_[b])
      if (view)
        encoder->useResource(view.get(), usage, stages);
  }
}

void MetalBindingVersionImpl::Bind(MTL::ComputeCommandEncoder *encoder, uint32_t group) const {
  encoder->setBuffer(arguments_.get(), 0, group);
  encoder->useResource(sizes_.get(), MTL::ResourceUsageRead);
  for (size_t b = 0; b < layout_.bindings.size(); ++b) {
    auto desc = layout_.bindings[b];
    uint32_t writable_texture = offsets_[b];
    if (desc.type == BINDING_RW_TEXTURE2D) {
      const bool tier1 = GetShaderCacheSettings().ios ||
                         device_->NativeDevice()->argumentBuffersSupport() == MTL::ArgumentBuffersTier1;
      for (const auto &view : native_views_[b]) {
        if (tier1) {
          auto texture = view.get();
          encoder->setTexture(texture, writable_texture);
        }
        ++writable_texture;
      }
    }
    if (desc.type == BINDING_SAMPLER)
      continue;  // Samplers are retained objects, not MTL::Resource residency entries.
    MTL::ResourceUsage usage = MTL::ResourceUsageRead;
    if (desc.type == BINDING_RW_STRUCTURED_BUFFER || desc.type == BINDING_RW_BYTE_ADDRESS_BUFFER ||
        desc.type == BINDING_RW_TEXTURE2D)
      usage |= MTL::ResourceUsageWrite;
    if (!(desc.visibility & SHADER_VISIBILITY_COMPUTE))
      continue;
    for (auto &resource : references_[b]) {
      MTL::Resource *native =
          desc.type == BINDING_ACCELERATION_STRUCTURE
              ? static_cast<MTL::Resource *>(rhi::QueryInterfacePtr<IMetalTopLevelAccelerationStructure>(resource.Get())
                                                 .Get()
                                                 ->NativeStructure())
          : desc.type <= BINDING_RW_BYTE_ADDRESS_BUFFER
              ? static_cast<MTL::Resource *>(rhi::QueryInterfacePtr<IMetalBuffer>(resource.Get()).Get()->NativeBuffer())
              : static_cast<MTL::Resource *>(rhi::QueryInterfacePtr<IMetalImage>(resource.Get()).Get()->NativeImage());
      encoder->useResource(native, usage);
    }
    for (const auto &view : native_views_[b])
      if (view)
        encoder->useResource(view.get(), usage);
  }
}

MetalBindingGroupImpl::MetalBindingGroupImpl(IMetalDevice *device, ObjectPtr<Version> version) {
  InitializeBindingGroup(device, ObjectPtr<IMetalBindingVersion>::Retain(version.Get()));
}

IBindingGroup *MetalBindingGroupImpl::Create(IMetalDevice *device, const BindingLayoutDesc &layout) {
  auto version = ObjectPtr<Version>::Adopt(Version::Create(device, layout));
  return version ? new MetalBindingGroupImpl(device, std::move(version)) : nullptr;
}

IMetalBindingVersion *IMetalBindingVersion::Create(IMetalDevice *device, const BindingLayoutDesc &layout) {
  return MetalBindingVersionImpl::Create(device, layout);
}

IBindingGroup *CreateMetalBindingGroup(IMetalDevice *device, const BindingLayoutDesc &layout) {
  return MetalBindingGroupImpl::Create(device, layout);
}

void MetalBindingVersionImpl::InitializeBindingVersion(const BindingLayoutDesc &layout) {
  layout_ = layout;

  std::sort(layout_.bindings.begin(), layout_.bindings.end(), [](auto a, auto b) { return a.binding < b.binding; });
  image_views_.resize(layout_.bindings.size());
  for (size_t i = 0; i < layout_.bindings.size(); ++i)
    image_views_[i].resize(layout_.bindings[i].count);
  native_views_.resize(layout_.bindings.size());
  for (size_t i = 0; i < layout_.bindings.size(); ++i)
    native_views_[i].resize(layout_.bindings[i].count);
  references_.resize(layout_.bindings.size());
  for (size_t i = 0; i < layout_.bindings.size(); ++i)
    references_[i].resize(layout_.bindings[i].count);
}

const BindingLayoutDesc &MetalBindingVersionImpl::GetLayout() const {
  return layout_;
}

const std::vector<std::vector<ObjectPtr<IObject>>> &MetalBindingVersionImpl::GetReferences() const {
  return references_;
}

const ImageView &MetalBindingVersionImpl::GetImageView(size_t binding, uint32_t element) const {
  return image_views_[binding][element];
}

bool MetalBindingVersionImpl::IsComplete() const {
  for (size_t i = 0; i < layout_.bindings.size(); ++i)
    for (uint32_t j = 0; j < layout_.bindings[i].count; ++j)
      if (!IsInitialized(i, j))
        return false;
  return true;
}

bool MetalBindingVersionImpl::Write(size_t binding_index,
                                    uint32_t first_element,
                                    const std::vector<BufferView> &values) {
  // Retain the entire update before releasing old slots, including resource swaps.
  std::vector<ObjectPtr<IObject>> references;
  references.reserve(values.size());
  for (const auto &value : values)
    references.push_back(ObjectPtr<IObject>::Retain(value.buffer));
  if (!WriteDescriptors(binding_index, first_element, values))
    return false;
  for (size_t i = 0; i < values.size(); ++i)
    references_[binding_index][first_element + i] = std::move(references[i]);
  return true;
}

bool MetalBindingVersionImpl::Write(size_t binding_index,
                                    uint32_t first_element,
                                    const std::vector<ImageView> &values) {
  // Retain the entire update before releasing old slots, including resource swaps.
  std::vector<ObjectPtr<IObject>> references;
  references.reserve(values.size());
  for (const auto &value : values)
    references.push_back(ObjectPtr<IObject>::Retain(value.image));
  if (!WriteDescriptors(binding_index, first_element, values))
    return false;
  for (size_t i = 0; i < values.size(); ++i)
    references_[binding_index][first_element + i] = std::move(references[i]);
  for (size_t i = 0; i < values.size(); ++i)
    image_views_[binding_index][first_element + i] = values[i];
  return true;
}

bool MetalBindingVersionImpl::Write(size_t binding_index,
                                    uint32_t first_element,
                                    const std::vector<ISampler *> &values) {
  std::vector<ObjectPtr<IObject>> references;
  references.reserve(values.size());
  for (auto value : values)
    references.push_back(ObjectPtr<IObject>::Retain(value));
  if (!WriteDescriptors(binding_index, first_element, values))
    return false;
  for (size_t i = 0; i < values.size(); ++i)
    references_[binding_index][first_element + i] = std::move(references[i]);
  return true;
}

bool MetalBindingVersionImpl::Write(size_t binding_index,
                                    uint32_t first_element,
                                    const std::vector<ITopLevelAccelerationStructure *> &values) {
  std::vector<ObjectPtr<IObject>> references;
  references.reserve(values.size());
  for (auto value : values)
    references.push_back(ObjectPtr<IObject>::Retain(value));
  if (!WriteDescriptors(binding_index, first_element, values))
    return false;
  for (size_t i = 0; i < values.size(); ++i)
    references_[binding_index][first_element + i] = std::move(references[i]);
  return true;
}

void MetalBindingGroupImpl::InitializeBindingGroup(IMetalDevice *device, ObjectPtr<IMetalBindingVersion> version) {
  device_ = device;
  layout_ = version->GetLayout();
  version_ = std::move(version);
}

const BindingLayoutDesc &MetalBindingGroupImpl::GetLayout() const {
  return layout_;
}

IMetalDevice *MetalBindingGroupImpl::Owner() const {
  return device_;
}

IMetalBindingVersion *MetalBindingGroupImpl::CurrentVersion() const {
  return version_.Get();
}

const BindingDesc *MetalBindingGroupImpl::FindBinding(uint32_t binding,
                                                      uint32_t first_element,
                                                      size_t count,
                                                      size_t &binding_index) const {
  for (binding_index = 0; binding_index < layout_.bindings.size(); ++binding_index) {
    const auto &desc = layout_.bindings[binding_index];
    if (desc.binding != binding)
      continue;
    if (!count || first_element > desc.count || count > desc.count - first_element)
      return nullptr;
    return &desc;
  }
  return nullptr;
}

bool MetalBindingGroupImpl::BindResource(uint32_t binding,
                                         const std::vector<BufferView> &buffers,
                                         uint32_t first_element) {
  size_t binding_index;
  const auto *desc = FindBinding(binding, first_element, buffers.size(), binding_index);
  if (!desc)
    return false;
  if (desc->type > BINDING_RW_BYTE_ADDRESS_BUFFER)
    return false;
  auto values = buffers;
  for (auto &view : values) {
#ifndef NDEBUG
    auto buffer = rhi::QueryInterfacePtr<IMetalBuffer>(view.buffer).Get();
    assert(!view.buffer || (buffer && buffer->Owner() == device_));
#endif
    if (!view.buffer || view.offset >= view.buffer->GetDesc().size)
      return false;
    if (!view.size)
      view.size = view.buffer->GetDesc().size - view.offset;
    if (view.size > view.buffer->GetDesc().size - view.offset)
      return false;
    if (desc->type == BINDING_STRUCTURED_BUFFER || desc->type == BINDING_RW_STRUCTURED_BUFFER) {
      if (view.offset % desc->stride || view.size % desc->stride)
        return false;
    } else if (desc->type == BINDING_BYTE_ADDRESS_BUFFER || desc->type == BINDING_RW_BYTE_ADDRESS_BUFFER) {
      if (view.offset % 4 || view.size % 4)
        return false;
    }
  }
  auto next = version_->IsUniquelyOwned() ? version_ : version_->Clone();
  if (!next)
    return false;
  if (!next->Write(binding_index, first_element, values))
    return false;
  version_ = std::move(next);
  return true;
}

bool MetalBindingGroupImpl::BindResource(uint32_t binding,
                                         const std::vector<ImageView> &images,
                                         uint32_t first_element) {
  size_t binding_index;
  const auto *desc = FindBinding(binding, first_element, images.size(), binding_index);
  if (!desc)
    return false;
  if (desc->type != BINDING_TEXTURE2D && desc->type != BINDING_RW_TEXTURE2D)
    return false;
  auto values = images;
  for (auto &view : values) {
    if (!NormalizeImageView(view, desc->type))
      return false;
    auto value = view.image;
    // Window frames have presentation ownership and cannot be captured as
    // texture bindings, matching the D3D12/Vulkan swapchain contract.
    if (rhi::QueryInterfacePtr<IMetalSwapChainImage>(value))
      return false;
    auto image = rhi::QueryInterfacePtr<IMetalImage>(value).Get();
    if (!image || !value->IsValid() || image->DeviceIdentity() != device_ || IsDepthFormat(value->GetDesc().format))
      return false;
    if (desc->type == BINDING_RW_TEXTURE2D && !image->SupportsStorage())
      return false;
  }
  auto next = version_->IsUniquelyOwned() ? version_ : version_->Clone();
  if (!next)
    return false;
  if (!next->Write(binding_index, first_element, values))
    return false;
  version_ = std::move(next);
  return true;
}

bool MetalBindingGroupImpl::BindResource(uint32_t binding,
                                         const std::vector<ISampler *> &samplers,
                                         uint32_t first_element) {
  size_t binding_index;
  const auto *desc = FindBinding(binding, first_element, samplers.size(), binding_index);
  if (!desc)
    return false;
  if (desc->type != BINDING_SAMPLER)
    return false;
  for (auto sampler : samplers) {
    auto native = rhi::QueryInterfacePtr<IMetalSampler>(sampler).Get();
    if (!native || native->DeviceIdentity() != device_)
      return false;
  }
  auto next = version_->IsUniquelyOwned() ? version_ : version_->Clone();
  if (!next)
    return false;
  if (!next->Write(binding_index, first_element, samplers))
    return false;
  version_ = std::move(next);
  return true;
}

bool MetalBindingGroupImpl::BindResource(uint32_t binding,
                                         const std::vector<ITopLevelAccelerationStructure *> &structures,
                                         uint32_t first_element) {
  size_t binding_index;
  auto desc = FindBinding(binding, first_element, structures.size(), binding_index);
  if (!desc || desc->type != BINDING_ACCELERATION_STRUCTURE)
    return false;
  for (auto structure : structures)
    if (!structure)
      return false;
  auto next = version_->IsUniquelyOwned() ? version_ : version_->Clone();
  if (!next)
    return false;
  if (!next->Write(binding_index, first_element, structures))
    return false;
  version_ = std::move(next);
  return true;
}
}  // namespace grassland::rhi::backend::metal
