#pragma once
#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12BindingGroup)

namespace grassland::rhi::backend::d3d12 {
class ID3D12BindingVersion : public IBackendBindingVersion {
 public:
  static ID3D12BindingVersion *Create(ID3D12Device *device, const BindingLayoutDesc &layout);
  virtual ObjectPtr<IBackendBindingVersion> Clone() const = 0;
  virtual bool IsInitialized(size_t binding_index, uint32_t element) const = 0;
  virtual uint32_t ResourceCount() const = 0;
  virtual D3D12DescriptorPool::Lease *ResourceDescriptors() const = 0;
  virtual D3D12DescriptorPool::Lease *SamplerDescriptors() const = 0;
};

class ID3D12BindingGroup : public IBackendBindingGroup {
 public:
  static ID3D12BindingGroup *Create(ID3D12Device *device, const BindingLayoutDesc &layout);
  using Version = ID3D12BindingVersion;
};

}  // namespace grassland::rhi::backend::d3d12

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12BindingVersion)
