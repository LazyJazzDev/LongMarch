#pragma once
#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12BindingVersion : public IBackendBindingVersion {
 public:
  static ID3D12BindingVersion *Create(ID3D12Device *device, const BindingLayoutDesc &layout);

  virtual uint32_t ResourceCount() const = 0;
  virtual D3D12DescriptorPool::Lease *ResourceDescriptors() const = 0;
  virtual D3D12DescriptorPool::Lease *SamplerDescriptors() const = 0;
};

IBindingGroup *CreateD3D12BindingGroup(ID3D12Device *device, const BindingLayoutDesc &layout);

}  // namespace grassland::rhi::backend::d3d12

LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::ID3D12BindingVersion, "1e82b239-9a3f-4e32-aeab-b2fdaba168ff")
