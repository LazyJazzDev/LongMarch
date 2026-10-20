#include "grassland/rhi/backend/metal/metal_binding_group.h"

#include <cstring>

#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_image.h"

namespace grassland::rhi::backend::metal {
uint32_t MetalArgumentCount(const BindingLayoutDesc &layout) {
  uint64_t count = 0;
  for (auto b : layout.bindings)
    count += b.count;
  // Bound allocation and index arithmetic before constructing native descriptors.
  return count < UINT32_MAX ? uint32_t(count) : 0;
}

bool ValidateMetalBindingLayout(MTL::Device *device, const BindingLayoutDesc &layout) {
  auto count = MetalArgumentCount(layout);
  if (!ValidateBindingLayout(layout) || (!layout.bindings.empty() && !count) ||
      device->argumentBuffersSupport() != MTL::ArgumentBuffersTier2 ||
      (uint64_t(count) + 1) * sizeof(uint64_t) > device->maxBufferLength())
    return false;
  uint64_t samplers = 0;
  for (auto b : layout.bindings)
    if (b.type == BINDING_SAMPLER)
      samplers += b.count;
  return samplers <= device->maxArgumentBufferSamplerCount();
}

MetalBindingGroup::Version::Version(MetalDevice *device, const BindingLayoutDesc &layout)
    : BindingGroup::Version(device, layout) {
}

MetalBindingGroup::Version *MetalBindingGroup::Version::Create(MetalDevice *device, const BindingLayoutDesc &layout) {
  if (!ValidateMetalBindingLayout(device->NativeDevice(), layout))
    return nullptr;
  try {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto result = std::unique_ptr<Version>(new Version(device, layout));
    std::vector<MTL::ArgumentDescriptor *> descriptors;
    uint32_t offset = 0;
    for (auto b : result->layout_.bindings) {
      auto d = MTL::ArgumentDescriptor::argumentDescriptor();
      d->setIndex(offset);
      d->setArrayLength(b.count);
      d->setDataType(b.type == BINDING_SAMPLER     ? MTL::DataTypeSampler
                     : b.type >= BINDING_TEXTURE2D ? MTL::DataTypeTexture
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
      result->samplers_.emplace_back(b.type == BINDING_SAMPLER ? b.count : 0);
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
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}

ObjectPtr<BindingGroup::Version> MetalBindingGroup::Version::Clone() const {
  auto copy = ObjectPtr<Version>::Adopt(Create(static_cast<MetalDevice *>(device_), layout_));
  if (!copy)
    return {};
  copy->references_ = references_;
  copy->initialized_ = initialized_;
  copy->samplers_ = samplers_;
  std::memcpy(copy->arguments_->contents(), arguments_->contents(), arguments_->length());
  std::memcpy(copy->sizes_->contents(), sizes_->contents(), sizes_->length());
  copy->encoder_->setBuffer(copy->sizes_.get(), 0, MetalArgumentCount(layout_));
  return ObjectPtr<BindingGroup::Version>::Retain(copy.Get());
}

bool MetalBindingGroup::Version::IsInitialized(size_t b, uint32_t e) const {
  return initialized_[b][e];
}

bool MetalBindingGroup::Version::WriteDescriptors(size_t b, uint32_t first, const std::vector<BufferView> &values) {
  for (auto v : values)
    if (v.offset % (layout_.bindings[b].type == BINDING_UNIFORM_BUFFER ? 256 : 4) || v.size > UINT32_MAX)
      return false;
  for (size_t i = 0; i < values.size(); ++i) {
    auto v = values[i];
    uint32_t index = offsets_[b] + first + i;
    encoder_->setBuffer(static_cast<MetalBuffer *>(v.buffer)->NativeBuffer(), v.offset, index);
    static_cast<uint32_t *>(sizes_->contents())[index] = uint32_t(v.size);
    initialized_[b][first + i] = true;
  }
  return true;
}

bool MetalBindingGroup::Version::WriteDescriptors(size_t b, uint32_t first, const std::vector<IImage *> &values) {
  for (size_t i = 0; i < values.size(); ++i) {
    encoder_->setTexture(static_cast<MetalImage *>(values[i])->NativeImage(), offsets_[b] + first + i);
    initialized_[b][first + i] = true;
  }
  return true;
}

bool MetalBindingGroup::Version::WriteDescriptors(size_t b, uint32_t first, const std::vector<SamplerDesc> &values) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  std::vector<NS::SharedPtr<MTL::SamplerState>> samplers;
  for (auto v : values) {
    auto d = NS::TransferPtr(MTL::SamplerDescriptor::alloc()->init());
    d->setSupportArgumentBuffers(true);
    d->setMinFilter(v.filter == SAMPLER_FILTER_NEAREST ? MTL::SamplerMinMagFilterNearest
                                                       : MTL::SamplerMinMagFilterLinear);
    d->setMagFilter(d->minFilter());
    const MTL::SamplerAddressMode modes[] = {MTL::SamplerAddressModeRepeat, MTL::SamplerAddressModeClampToEdge,
                                             MTL::SamplerAddressModeMirrorRepeat};
    d->setSAddressMode(modes[v.address_u]);
    d->setTAddressMode(modes[v.address_v]);
    auto sampler = NS::TransferPtr(static_cast<MetalDevice *>(device_)->NativeDevice()->newSamplerState(d.get()));
    if (!sampler)
      return false;
    samplers.push_back(std::move(sampler));
  }
  for (size_t i = 0; i < values.size(); ++i) {
    encoder_->setSamplerState(samplers[i].get(), offsets_[b] + first + i);
    samplers_[b][first + i] = std::move(samplers[i]);
    initialized_[b][first + i] = true;
  }
  return true;
}

void MetalBindingGroup::Version::Bind(MTL::RenderCommandEncoder *encoder, uint32_t group) const {
  encoder->setVertexBuffer(arguments_.get(), 0, group);
  encoder->setFragmentBuffer(arguments_.get(), 0, group);
  encoder->useResource(sizes_.get(), MTL::ResourceUsageRead, MTL::RenderStageVertex | MTL::RenderStageFragment);
  for (size_t b = 0; b < layout_.bindings.size(); ++b) {
    auto desc = layout_.bindings[b];
    MTL::ResourceUsage usage = MTL::ResourceUsageRead;
    if (desc.type == BINDING_RW_STRUCTURED_BUFFER || desc.type == BINDING_RW_BYTE_ADDRESS_BUFFER ||
        desc.type == BINDING_RW_TEXTURE2D)
      usage |= MTL::ResourceUsageWrite;
    MTL::RenderStages stages = 0;
    if (desc.visibility & SHADER_VISIBILITY_VERTEX)
      stages |= MTL::RenderStageVertex;
    if (desc.visibility & SHADER_VISIBILITY_FRAGMENT)
      stages |= MTL::RenderStageFragment;
    for (auto &resource : references_[b]) {
      MTL::Resource *native =
          desc.type <= BINDING_RW_BYTE_ADDRESS_BUFFER
              ? static_cast<MTL::Resource *>(static_cast<MetalBuffer *>(resource.Get())->NativeBuffer())
              : static_cast<MTL::Resource *>(static_cast<MetalImage *>(resource.Get())->NativeImage());
      encoder->useResource(native, usage, stages);
    }
  }
}

MetalBindingGroup::MetalBindingGroup(MetalDevice *device, ObjectPtr<Version> version)
    : BindingGroup(device, ObjectPtr<BindingGroup::Version>::Retain(version.Get())) {
}

MetalBindingGroup *MetalBindingGroup::Create(MetalDevice *device, const BindingLayoutDesc &layout) {
  try {
    auto version = ObjectPtr<Version>::Adopt(Version::Create(device, layout));
    return version ? new MetalBindingGroup(device, std::move(version)) : nullptr;
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}
}  // namespace grassland::rhi::backend::metal
