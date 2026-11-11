#include "grassland/rhi/backend/d3d12/d3d12_binding_group.h"

#include <algorithm>
#include <set>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_buffer.h"
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/backend_sampler.h"
#include "grassland/rhi/backend/d3d12/d3d12_acceleration_structure.h"
#include "grassland/rhi/backend/d3d12/d3d12_allocated_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_binding_group_impl.h"
#include "grassland/rhi/backend/d3d12/d3d12_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_sampler.h"
#include "grassland/rhi/rhi_acceleration_structure.h"

namespace grassland::rhi::backend::d3d12 {
D3D12BindingVersionImpl::D3D12BindingVersionImpl(ID3D12Device *device, const BindingLayoutDesc &layout)
    : device_(device) {
  InitializeBindingVersion(layout);

  uint32_t resource = 0, sampler = 0;
  for (size_t i = 0; i < layout_.bindings.size(); ++i) {
    const auto &binding = layout_.bindings[i];
    auto &offset = binding.type == BINDING_SAMPLER ? sampler : resource;
    offsets_.push_back(offset);
    offset += binding.count;
    initialized_.emplace_back(binding.count, false);
  }
}

D3D12BindingVersionImpl *D3D12BindingVersionImpl::Create(ID3D12Device *device, const BindingLayoutDesc &layout) {
  uint64_t resources = 0, samplers = 0;
  for (auto b : layout.bindings) {
    if ((b.type == BINDING_STRUCTURED_BUFFER || b.type == BINDING_RW_STRUCTURED_BUFFER) &&
        (b.stride > 2048 || b.stride % 4))
      return nullptr;
    (b.type == BINDING_SAMPLER ? samplers : resources) += b.count;
  }
  if (resources > 1000000 || samplers > 2048)
    return nullptr;
  auto version = std::unique_ptr<D3D12BindingVersionImpl>(new D3D12BindingVersionImpl(device, layout));
  if (resources) {
    version->resources_ = device->ResourceDescriptorStorage().Acquire(uint32_t(resources));
    if (!version->resources_)
      return nullptr;
  }
  if (samplers) {
    version->samplers_ = device->SamplerDescriptorStorage().Acquire(uint32_t(samplers));
    if (!version->samplers_)
      return nullptr;
  }
  return version.release();
}

ObjectPtr<IBackendBindingVersion> D3D12BindingVersionImpl::Clone() const {
  auto copy = ObjectPtr<D3D12BindingVersionImpl>::Adopt(Create(device_, layout_));
  if (!copy)
    return {};
  copy->references_ = references_;
  copy->initialized_ = initialized_;
  auto native = device_->NativeDevice();
  for (size_t i = 0; i < layout_.bindings.size(); ++i) {
    const bool sampler = layout_.bindings[i].type == BINDING_SAMPLER;
    auto source = sampler ? samplers_.Get() : resources_.Get();
    auto target = sampler ? copy->samplers_.Get() : copy->resources_.Get();
    const auto type = sampler ? D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER : D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    for (uint32_t j = 0; j < layout_.bindings[i].count; ++j)
      if (initialized_[i][j])
        native->CopyDescriptorsSimple(1, target->CPU(offsets_[i] + j), source->CPU(offsets_[i] + j), type);
  }
  return copy;
}

bool D3D12BindingVersionImpl::IsInitialized(size_t binding_index, uint32_t element) const {
  return initialized_[binding_index][element];
}

uint32_t D3D12BindingVersionImpl::ResourceCount() const {
  return resources_ ? resources_->Count() : 0;
}

D3D12DescriptorPool::Lease *D3D12BindingVersionImpl::SamplerDescriptors() const {
  return samplers_.Get();
}

bool D3D12BindingVersionImpl::WriteDescriptors(size_t binding_index,
                                               uint32_t first_element,
                                               const std::vector<BufferView> &buffers) {
  const auto &binding = layout_.bindings[binding_index];
  if (binding.type == BINDING_UNIFORM_BUFFER) {
    for (const auto &view : buffers) {
      if (view.size > 65536)
        return false;
      const auto size = (view.size + 255) & ~uint64_t(255);
      if (view.offset % 256 || size > view.buffer->GetDesc().size - view.offset)
        return false;
    }
  }
  auto native = device_->NativeDevice();
  switch (binding.type) {
    case BINDING_UNIFORM_BUFFER:
      for (size_t i = 0; i < buffers.size(); ++i) {
        const auto &view = buffers[i];
        auto buffer = rhi::QueryInterfacePtr<ID3D12Buffer>(view.buffer).Get();
        D3D12_CONSTANT_BUFFER_VIEW_DESC desc{buffer->NativeBuffer()->GetGPUVirtualAddress() + view.offset,
                                             UINT((view.size + 255) & ~uint64_t(255))};
        native->CreateConstantBufferView(&desc, resources_->CPU(offsets_[binding_index] + first_element + uint32_t(i)));
      }
      break;
    case BINDING_STRUCTURED_BUFFER:
    case BINDING_BYTE_ADDRESS_BUFFER: {
      const bool raw = binding.type == BINDING_BYTE_ADDRESS_BUFFER;
      const uint32_t stride = raw ? 4 : binding.stride;
      for (const auto &view : buffers)
        if (view.size / stride > UINT32_MAX)
          return false;
      for (size_t i = 0; i < buffers.size(); ++i) {
        const auto &view = buffers[i];
        D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
        desc.Format = raw ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_UNKNOWN;
        desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc.Buffer.FirstElement = view.offset / stride;
        desc.Buffer.NumElements = UINT(view.size / stride);
        desc.Buffer.StructureByteStride = raw ? 0 : stride;
        desc.Buffer.Flags = raw ? D3D12_BUFFER_SRV_FLAG_RAW : D3D12_BUFFER_SRV_FLAG_NONE;
        native->CreateShaderResourceView(rhi::QueryInterfacePtr<ID3D12Buffer>(view.buffer).Get()->NativeBuffer(), &desc,
                                         resources_->CPU(offsets_[binding_index] + first_element + uint32_t(i)));
      }
      break;
    }
    case BINDING_RW_STRUCTURED_BUFFER:
    case BINDING_RW_BYTE_ADDRESS_BUFFER: {
      const bool raw = binding.type == BINDING_RW_BYTE_ADDRESS_BUFFER;
      const uint32_t stride = raw ? 4 : binding.stride;
      for (const auto &view : buffers)
        if (view.size / stride > UINT32_MAX)
          return false;
      for (size_t i = 0; i < buffers.size(); ++i) {
        const auto &view = buffers[i];
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
        desc.Format = raw ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_UNKNOWN;
        desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        desc.Buffer.FirstElement = view.offset / stride;
        desc.Buffer.NumElements = UINT(view.size / stride);
        desc.Buffer.StructureByteStride = raw ? 0 : stride;
        desc.Buffer.Flags = raw ? D3D12_BUFFER_UAV_FLAG_RAW : D3D12_BUFFER_UAV_FLAG_NONE;
        native->CreateUnorderedAccessView(rhi::QueryInterfacePtr<ID3D12Buffer>(view.buffer).Get()->NativeBuffer(),
                                          nullptr, &desc,
                                          resources_->CPU(offsets_[binding_index] + first_element + uint32_t(i)));
      }
      break;
    }
    default:
      return false;
  }
  for (size_t i = 0; i < buffers.size(); ++i)
    initialized_[binding_index][first_element + i] = true;
  return true;
}

bool D3D12BindingVersionImpl::WriteDescriptors(size_t binding_index,
                                               uint32_t first_element,
                                               const std::vector<IImage *> &images) {
  const auto &binding = layout_.bindings[binding_index];
  for (auto image : images)
    if (!rhi::QueryInterfacePtr<ID3D12Image>(image).Get())
      return false;
  auto native = device_->NativeDevice();
  if (binding.type == BINDING_TEXTURE2D) {
    for (size_t i = 0; i < images.size(); ++i)
      native->CreateShaderResourceView(rhi::QueryInterfacePtr<ID3D12Image>(images[i]).Get()->NativeImage(), nullptr,
                                       resources_->CPU(offsets_[binding_index] + first_element + uint32_t(i)));
  } else {
    for (size_t i = 0; i < images.size(); ++i)
      native->CreateUnorderedAccessView(rhi::QueryInterfacePtr<ID3D12Image>(images[i]).Get()->NativeImage(), nullptr,
                                        nullptr,
                                        resources_->CPU(offsets_[binding_index] + first_element + uint32_t(i)));
  }
  for (size_t i = 0; i < images.size(); ++i)
    initialized_[binding_index][first_element + i] = true;
  return true;
}

bool D3D12BindingVersionImpl::WriteDescriptors(size_t binding_index,
                                               uint32_t first_element,
                                               const std::vector<ISampler *> &samplers) {
  auto native = device_->NativeDevice();
  for (size_t i = 0; i < samplers.size(); ++i)
    native->CopyDescriptorsSimple(1, samplers_->CPU(offsets_[binding_index] + first_element + uint32_t(i)),
                                  rhi::QueryInterfacePtr<ID3D12Sampler>(samplers[i]).Get()->Descriptor(),
                                  D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
  for (size_t i = 0; i < samplers.size(); ++i)
    initialized_[binding_index][first_element + i] = true;
  return true;
}

D3D12DescriptorPool::Lease *D3D12BindingVersionImpl::ResourceDescriptors() const {
  return resources_.Get();
}

D3D12BindingGroupImpl::D3D12BindingGroupImpl(ID3D12Device *device, ObjectPtr<Version> version) {
  InitializeBindingGroup(device, std::move(version));
}

IBindingGroup *D3D12BindingGroupImpl::Create(ID3D12Device *device, const BindingLayoutDesc &layout) {
  if (!ValidateBindingLayout(layout))
    return nullptr;
  auto version = ObjectPtr<Version>::Adopt(Version::Create(device, layout));
  if (!version)
    return nullptr;
  return new D3D12BindingGroupImpl(device, std::move(version));
}

bool D3D12BindingVersionImpl::WriteDescriptors(size_t binding_index,
                                               uint32_t first_element,
                                               const std::vector<ITopLevelAccelerationStructure *> &values) {
  for (auto value : values) {
    auto native = rhi::QueryInterfacePtr<ID3D12TopLevelAccelerationStructure>(value).Get();
    if (!native || native->Owner() != device_)
      return false;
  }
  auto device = device_->NativeDevice();
  for (size_t i = 0; i < values.size(); ++i) {
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc.RaytracingAccelerationStructure.Location =
        rhi::QueryInterfacePtr<ID3D12TopLevelAccelerationStructure>(values[i]).Get()->Address();
    device->CreateShaderResourceView(nullptr, &desc,
                                     resources_->CPU(offsets_[binding_index] + first_element + uint32_t(i)));
  }
  for (size_t i = 0; i < values.size(); ++i)
    initialized_[binding_index][first_element + i] = true;
  return true;
}

ID3D12BindingVersion *ID3D12BindingVersion::Create(ID3D12Device *device, const BindingLayoutDesc &layout) {
  return D3D12BindingVersionImpl::Create(device, layout);
}

IBindingGroup *CreateD3D12BindingGroup(ID3D12Device *device, const BindingLayoutDesc &layout) {
  return D3D12BindingGroupImpl::Create(device, layout);
}

void D3D12BindingVersionImpl::InitializeBindingVersion(const BindingLayoutDesc &layout) {
  layout_ = layout;

  std::sort(layout_.bindings.begin(), layout_.bindings.end(), [](auto a, auto b) { return a.binding < b.binding; });
  references_.resize(layout_.bindings.size());
  for (size_t i = 0; i < layout_.bindings.size(); ++i)
    references_[i].resize(layout_.bindings[i].count);
}

const BindingLayoutDesc &D3D12BindingVersionImpl::GetLayout() const {
  return layout_;
}

const std::vector<std::vector<ObjectPtr<IObject>>> &D3D12BindingVersionImpl::GetReferences() const {
  return references_;
}

bool D3D12BindingVersionImpl::IsComplete() const {
  for (size_t i = 0; i < layout_.bindings.size(); ++i)
    for (uint32_t j = 0; j < layout_.bindings[i].count; ++j)
      if (!IsInitialized(i, j))
        return false;
  return true;
}

bool D3D12BindingVersionImpl::Write(size_t binding_index,
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

bool D3D12BindingVersionImpl::Write(size_t binding_index, uint32_t first_element, const std::vector<IImage *> &values) {
  // Retain the entire update before releasing old slots, including resource swaps.
  std::vector<ObjectPtr<IObject>> references;
  references.reserve(values.size());
  for (const auto &value : values)
    references.push_back(ObjectPtr<IObject>::Retain(value));
  if (!WriteDescriptors(binding_index, first_element, values))
    return false;
  for (size_t i = 0; i < values.size(); ++i)
    references_[binding_index][first_element + i] = std::move(references[i]);
  return true;
}

bool D3D12BindingVersionImpl::Write(size_t binding_index,
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

bool D3D12BindingVersionImpl::Write(size_t binding_index,
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

void D3D12BindingGroupImpl::InitializeBindingGroup(IBackendDevice *device, ObjectPtr<IBackendBindingVersion> version) {
  device_ = device;
  layout_ = version->GetLayout();
  version_ = std::move(version);
}

const BindingLayoutDesc &D3D12BindingGroupImpl::GetLayout() const {
  return layout_;
}

IBackendDevice *D3D12BindingGroupImpl::Owner() const {
  return device_;
}

IBackendBindingVersion *D3D12BindingGroupImpl::CurrentVersion() const {
  return version_.Get();
}

const BindingDesc *D3D12BindingGroupImpl::FindBinding(uint32_t binding,
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

bool D3D12BindingGroupImpl::BindResource(uint32_t binding,
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
    auto buffer = rhi::QueryInterfacePtr<IBackendBuffer>(view.buffer).Get();
    if (!buffer || buffer->Owner() != device_ || view.offset >= view.buffer->GetDesc().size)
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

bool D3D12BindingGroupImpl::BindResource(uint32_t binding,
                                         const std::vector<IImage *> &images,
                                         uint32_t first_element) {
  size_t binding_index;
  const auto *desc = FindBinding(binding, first_element, images.size(), binding_index);
  if (!desc)
    return false;
  if (desc->type != BINDING_TEXTURE2D && desc->type != BINDING_RW_TEXTURE2D)
    return false;
  for (auto value : images) {
    auto image = rhi::QueryInterfacePtr<IBackendImage>(value).Get();
    if (!image || !value->IsValid() || image->DeviceIdentity() != device_ || IsDepthFormat(value->GetDesc().format))
      return false;
    if (desc->type == BINDING_RW_TEXTURE2D && !image->SupportsStorage())
      return false;
  }
  auto next = version_->IsUniquelyOwned() ? version_ : version_->Clone();
  if (!next)
    return false;
  if (!next->Write(binding_index, first_element, images))
    return false;
  version_ = std::move(next);
  return true;
}

bool D3D12BindingGroupImpl::BindResource(uint32_t binding,
                                         const std::vector<ISampler *> &samplers,
                                         uint32_t first_element) {
  size_t binding_index;
  const auto *desc = FindBinding(binding, first_element, samplers.size(), binding_index);
  if (!desc)
    return false;
  if (desc->type != BINDING_SAMPLER)
    return false;
  for (auto sampler : samplers) {
    auto native = rhi::QueryInterfacePtr<IBackendSampler>(sampler).Get();
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

bool D3D12BindingGroupImpl::BindResource(uint32_t binding,
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
}  // namespace grassland::rhi::backend::d3d12

LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::ID3D12BindingVersion, "1e82b239-9a3f-4e32-aeab-b2fdaba168ff")
