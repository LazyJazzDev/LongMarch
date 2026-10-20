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
  for (auto b : layout_.bindings)
    for (uint32_t i = 0; i < b.count; ++i) {
      types_.push_back(b.type);
      offsets_.push_back(b.type == BINDING_SAMPLER ? sampler++ : resource++);
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

ObjectPtr<BindingGroup::Version> D3D12BindingGroup::Version::CreateEmpty() const {
  return ObjectPtr<Version>::Adopt(Create(static_cast<D3D12Device *>(device_), layout_));
}

uint32_t D3D12BindingGroup::Version::ResourceCount() const {
  return resources_ ? resources_->Count() : 0;
}

D3D12DescriptorPool::Lease *D3D12BindingGroup::Version::SamplerDescriptors() const {
  return samplers_.Get();
}

bool D3D12BindingGroup::Version::WriteDescriptors(size_t offset, const std::vector<Resource> &resources) {
  for (size_t i = 0; i < resources.size(); ++i) {
    const auto &e = resources[i];
    if (e.image && !dynamic_cast<D3D12AllocatedImage *>(e.image))
      return false;
    if (types_[offset + i] == BINDING_UNIFORM_BUFFER) {
      if (e.size > 65536)
        return false;
      const auto size = (e.size + 255) & ~uint64_t(255);
      if (e.offset % 256 || size > e.buffer->GetDesc().size - e.offset)
        return false;
    }
  }
  // All fallible checks precede mutation. This version is unreferenced by commands.
  resource_variants_.clear();
  auto native = static_cast<D3D12Device *>(device_)->NativeDevice();
  for (size_t i = 0; i < resources.size(); ++i) {
    const auto &e = resources[i];
    auto type = types_[offset + i];
    auto handle = (type == BINDING_SAMPLER ? samplers_ : resources_)->CPU(offsets_[offset + i]);
    if (type == BINDING_UNIFORM_BUFFER) {
      auto buffer = static_cast<D3D12Buffer *>(e.buffer);
      D3D12_CONSTANT_BUFFER_VIEW_DESC desc{buffer->NativeBuffer()->GetGPUVirtualAddress() + e.offset,
                                           UINT((e.size + 255) & ~uint64_t(255))};
      native->CreateConstantBufferView(&desc, handle);
    } else if (type == BINDING_STORAGE_BUFFER) {
      // A valid placeholder for bindings optimized out by a Program. Active
      // structured views are materialized once their reflected stride is known.
      D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
      desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
      desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      desc.Buffer.NumElements = 1;
      desc.Buffer.StructureByteStride = 4;
      native->CreateShaderResourceView(nullptr, &desc, handle);
    } else if (type == BINDING_RW_STORAGE_BUFFER) {
      D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
      desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
      desc.Buffer.NumElements = 1;
      desc.Buffer.StructureByteStride = 4;
      native->CreateUnorderedAccessView(nullptr, nullptr, &desc, handle);
    } else if (type == BINDING_TEXTURE2D) {
      native->CreateShaderResourceView(static_cast<D3D12Image *>(e.image)->NativeImage(), nullptr, handle);
    } else if (type == BINDING_RW_TEXTURE2D) {
      native->CreateUnorderedAccessView(static_cast<D3D12Image *>(e.image)->NativeImage(), nullptr, nullptr, handle);
    } else {
      D3D12_SAMPLER_DESC desc{};
      desc.Filter =
          e.sampler.filter == SAMPLER_FILTER_LINEAR ? D3D12_FILTER_MIN_MAG_MIP_LINEAR : D3D12_FILTER_MIN_MAG_MIP_POINT;
      const D3D12_TEXTURE_ADDRESS_MODE modes[] = {D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                                                  D3D12_TEXTURE_ADDRESS_MODE_MIRROR};
      desc.AddressU = modes[e.sampler.address_u];
      desc.AddressV = desc.AddressW = modes[e.sampler.address_v];
      desc.MaxLOD = D3D12_FLOAT32_MAX;
      desc.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
      desc.MaxAnisotropy = 1;
      native->CreateSampler(&desc, handle);
    }
  }
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
  for (const auto &e : GetResources()) {
    auto stride = program->StructuredBufferStride(group, e.binding);
    if (stride && (e.offset % stride || e.size % stride || e.size / stride > UINT32_MAX))
      return nullptr;
  }
  auto device = static_cast<D3D12Device *>(device_);
  auto table = device->ResourceDescriptorStorage().Acquire(resources_->Count());
  if (!table)
    return nullptr;
  auto native = device->NativeDevice();
  native->CopyDescriptorsSimple(resources_->Count(), table->CPU(), resources_->CPU(),
                                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  for (size_t i = 0; i < GetResources().size(); ++i) {
    const auto &e = GetResources()[i];
    const auto stride = program->StructuredBufferStride(group, e.binding);
    if (!stride)
      continue;
    auto buffer = static_cast<D3D12Buffer *>(e.buffer)->NativeBuffer();
    auto handle = table->CPU(offsets_[i]);
    if (types_[i] == BINDING_STORAGE_BUFFER) {
      D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
      desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
      desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      desc.Buffer.FirstElement = e.offset / stride;
      desc.Buffer.NumElements = UINT(e.size / stride);
      desc.Buffer.StructureByteStride = stride;
      native->CreateShaderResourceView(buffer, &desc, handle);
    } else {
      D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
      desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
      desc.Buffer.FirstElement = e.offset / stride;
      desc.Buffer.NumElements = UINT(e.size / stride);
      desc.Buffer.StructureByteStride = stride;
      native->CreateUnorderedAccessView(buffer, nullptr, &desc, handle);
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
