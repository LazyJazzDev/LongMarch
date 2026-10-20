#include "grassland/rhi/backend/d3d12/d3d12_binding_group.h"

#include <algorithm>

#include "grassland/rhi/backend/d3d12/d3d12_allocated_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_program.h"

namespace grassland::rhi::backend::d3d12 {
D3D12BindingGroup::Version::Version(D3D12Device *device, const BindingLayoutDesc &layout)
    : BindingGroup::Version(device, layout) {
  uint32_t resource = 0, sampler = 0;
  structured_ranges_.resize(layout_.bindings.size());
  for (size_t i = 0; i < layout_.bindings.size(); ++i) {
    const auto &binding = layout_.bindings[i];
    auto &offset = binding.type == BINDING_SAMPLER ? sampler : resource;
    offsets_.push_back(offset);
    offset += binding.count;
    initialized_.emplace_back(binding.count, false);
    if (binding.type == BINDING_STORAGE_BUFFER || binding.type == BINDING_RW_STORAGE_BUFFER)
      structured_ranges_[i].resize(binding.count);
  }
}

D3D12BindingGroup::Version *D3D12BindingGroup::Version::Create(D3D12Device *device, const BindingLayoutDesc &layout) {
  uint64_t resources = 0, samplers = 0;
  for (auto b : layout.bindings)
    (b.type == BINDING_SAMPLER ? samplers : resources) += b.count;
  if (resources > 1000000 || samplers > 2048)
    return nullptr;
  auto version = std::unique_ptr<Version>(new Version(device, layout));
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

ObjectPtr<BindingGroup::Version> D3D12BindingGroup::Version::Clone() const {
  auto device = static_cast<D3D12Device *>(device_);
  auto copy = ObjectPtr<Version>::Adopt(Create(device, layout_));
  if (!copy)
    return {};
  copy->references_ = references_;
  copy->structured_ranges_ = structured_ranges_;
  copy->initialized_ = initialized_;
  auto native = device->NativeDevice();
  for (size_t i = 0; i < layout_.bindings.size(); ++i) {
    const bool sampler = layout_.bindings[i].type == BINDING_SAMPLER;
    auto source = sampler ? samplers_.Get() : resources_.Get();
    auto target = sampler ? copy->samplers_.Get() : copy->resources_.Get();
    const auto type = sampler ? D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER : D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    for (uint32_t j = 0; j < layout_.bindings[i].count; ++j)
      if (initialized_[i][j])
        native->CopyDescriptorsSimple(1, target->CPU(offsets_[i] + j), source->CPU(offsets_[i] + j), type);
  }
  // Stride-specific tables are immutable; subsequent writes drop only the new version's cache.
  copy->resource_variants_ = resource_variants_;
  return copy;
}

bool D3D12BindingGroup::Version::IsInitialized(size_t binding_index, uint32_t element) const {
  return initialized_[binding_index][element];
}

uint32_t D3D12BindingGroup::Version::ResourceCount() const {
  return resources_ ? resources_->Count() : 0;
}

D3D12DescriptorPool::Lease *D3D12BindingGroup::Version::SamplerDescriptors() const {
  return samplers_.Get();
}

bool D3D12BindingGroup::Version::WriteDescriptors(size_t binding_index,
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
  resource_variants_.clear();
  auto native = static_cast<D3D12Device *>(device_)->NativeDevice();
  switch (binding.type) {
    case BINDING_UNIFORM_BUFFER:
      for (size_t i = 0; i < buffers.size(); ++i) {
        const auto &view = buffers[i];
        auto buffer = static_cast<D3D12Buffer *>(view.buffer);
        D3D12_CONSTANT_BUFFER_VIEW_DESC desc{buffer->NativeBuffer()->GetGPUVirtualAddress() + view.offset,
                                             UINT((view.size + 255) & ~uint64_t(255))};
        native->CreateConstantBufferView(&desc, resources_->CPU(offsets_[binding_index] + first_element + uint32_t(i)));
      }
      break;
    case BINDING_STORAGE_BUFFER: {
      // Active structured views are materialized when the Program's stride is known.
      D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
      desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
      desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      desc.Buffer.NumElements = 1;
      desc.Buffer.StructureByteStride = 4;
      for (size_t i = 0; i < buffers.size(); ++i)
        native->CreateShaderResourceView(nullptr, &desc,
                                         resources_->CPU(offsets_[binding_index] + first_element + uint32_t(i)));
      break;
    }
    case BINDING_RW_STORAGE_BUFFER: {
      D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
      desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
      desc.Buffer.NumElements = 1;
      desc.Buffer.StructureByteStride = 4;
      for (size_t i = 0; i < buffers.size(); ++i)
        native->CreateUnorderedAccessView(nullptr, nullptr, &desc,
                                          resources_->CPU(offsets_[binding_index] + first_element + uint32_t(i)));
      break;
    }
    default:
      return false;
  }
  for (size_t i = 0; i < buffers.size(); ++i)
    initialized_[binding_index][first_element + i] = true;
  if (!structured_ranges_[binding_index].empty())
    for (size_t i = 0; i < buffers.size(); ++i)
      structured_ranges_[binding_index][first_element + i] = {buffers[i].offset, buffers[i].size};
  return true;
}

bool D3D12BindingGroup::Version::WriteDescriptors(size_t binding_index,
                                                  uint32_t first_element,
                                                  const std::vector<IImage *> &images) {
  const auto &binding = layout_.bindings[binding_index];
  for (auto image : images)
    if (!dynamic_cast<D3D12AllocatedImage *>(image))
      return false;
  resource_variants_.clear();
  auto native = static_cast<D3D12Device *>(device_)->NativeDevice();
  if (binding.type == BINDING_TEXTURE2D) {
    for (size_t i = 0; i < images.size(); ++i)
      native->CreateShaderResourceView(static_cast<D3D12Image *>(images[i])->NativeImage(), nullptr,
                                       resources_->CPU(offsets_[binding_index] + first_element + uint32_t(i)));
  } else {
    for (size_t i = 0; i < images.size(); ++i)
      native->CreateUnorderedAccessView(static_cast<D3D12Image *>(images[i])->NativeImage(), nullptr, nullptr,
                                        resources_->CPU(offsets_[binding_index] + first_element + uint32_t(i)));
  }
  for (size_t i = 0; i < images.size(); ++i)
    initialized_[binding_index][first_element + i] = true;
  return true;
}

bool D3D12BindingGroup::Version::WriteDescriptors(size_t binding_index,
                                                  uint32_t first_element,
                                                  const std::vector<SamplerDesc> &samplers) {
  auto native = static_cast<D3D12Device *>(device_)->NativeDevice();
  const D3D12_TEXTURE_ADDRESS_MODE modes[] = {D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                                              D3D12_TEXTURE_ADDRESS_MODE_MIRROR};
  for (size_t i = 0; i < samplers.size(); ++i) {
    const auto &sampler = samplers[i];
    D3D12_SAMPLER_DESC desc{};
    desc.Filter =
        sampler.filter == SAMPLER_FILTER_LINEAR ? D3D12_FILTER_MIN_MAG_MIP_LINEAR : D3D12_FILTER_MIN_MAG_MIP_POINT;
    desc.AddressU = modes[sampler.address_u];
    desc.AddressV = desc.AddressW = modes[sampler.address_v];
    desc.MaxLOD = D3D12_FLOAT32_MAX;
    desc.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    desc.MaxAnisotropy = 1;
    native->CreateSampler(&desc, samplers_->CPU(offsets_[binding_index] + first_element + uint32_t(i)));
  }
  for (size_t i = 0; i < samplers.size(); ++i)
    initialized_[binding_index][first_element + i] = true;
  return true;
}

D3D12DescriptorPool::Lease *D3D12BindingGroup::Version::ResourceDescriptors(const D3D12Program *program,
                                                                            uint32_t group) {
  if (!resources_)
    return nullptr;
  std::vector<uint32_t> strides;
  for (auto b : layout_.bindings)
    if (b.type == BINDING_STORAGE_BUFFER || b.type == BINDING_RW_STORAGE_BUFFER)
      strides.push_back(program->StructuredBufferStride(group, b.binding));
  if (std::all_of(strides.begin(), strides.end(), [](auto stride) { return stride == 0; }))
    return resources_.Get();
  for (const auto &variant : resource_variants_)
    if (variant.first == strides)
      return variant.second.Get();
  for (size_t i = 0; i < layout_.bindings.size(); ++i) {
    const auto stride = program->StructuredBufferStride(group, layout_.bindings[i].binding);
    if (!stride)
      continue;
    for (const auto &[offset, size] : structured_ranges_[i])
      if (offset % stride || size % stride || size / stride > UINT32_MAX)
        return nullptr;
  }
  auto device = static_cast<D3D12Device *>(device_);
  auto table = device->ResourceDescriptorStorage().Acquire(resources_->Count());
  if (!table)
    return nullptr;
  auto native = device->NativeDevice();
  native->CopyDescriptorsSimple(resources_->Count(), table->CPU(), resources_->CPU(),
                                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  for (size_t i = 0; i < layout_.bindings.size(); ++i) {
    const auto &binding = layout_.bindings[i];
    const auto stride = program->StructuredBufferStride(group, binding.binding);
    if (!stride)
      continue;
    for (uint32_t j = 0; j < binding.count; ++j) {
      const auto [offset, size] = structured_ranges_[i][j];
      auto buffer = static_cast<D3D12Buffer *>(references_[i][j].Get())->NativeBuffer();
      auto handle = table->CPU(offsets_[i] + j);
      if (binding.type == BINDING_STORAGE_BUFFER) {
        D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
        desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc.Buffer.FirstElement = offset / stride;
        desc.Buffer.NumElements = UINT(size / stride);
        desc.Buffer.StructureByteStride = stride;
        native->CreateShaderResourceView(buffer, &desc, handle);
      } else {
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
        desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        desc.Buffer.FirstElement = offset / stride;
        desc.Buffer.NumElements = UINT(size / stride);
        desc.Buffer.StructureByteStride = stride;
        native->CreateUnorderedAccessView(buffer, nullptr, &desc, handle);
      }
    }
  }
  auto result = table.Get();
  resource_variants_.emplace_back(std::move(strides), std::move(table));
  return result;
}

D3D12BindingGroup::D3D12BindingGroup(D3D12Device *device, ObjectPtr<Version> version)
    : BindingGroup(device, std::move(version)) {
}

D3D12BindingGroup *D3D12BindingGroup::Create(D3D12Device *device, const BindingLayoutDesc &layout) {
  try {
    if (!ValidateBindingLayout(layout))
      return nullptr;
    auto version = ObjectPtr<Version>::Adopt(Version::Create(device, layout));
    if (!version)
      return nullptr;
    return new D3D12BindingGroup(device, std::move(version));
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}
}  // namespace grassland::rhi::backend::d3d12
