#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12BindingGroup;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::D3D12BindingGroup)
#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Device;

class D3D12BindingGroup final : public BindingGroup {
 public:
  LONGMARCH_RHI_QUERY(D3D12BindingGroup, BindingGroup)

  static D3D12BindingGroup *Create(D3D12Device *device, const BindingLayoutDesc &layout);

  class Version final : public BindingGroup::Version {
   public:
    using BindingGroup::Version::QueryInterface;
    bool QueryInterface(InterfaceID id, void **output) noexcept override;

    static Version *Create(D3D12Device *device, const BindingLayoutDesc &layout);
    ObjectPtr<BindingGroup::Version> Clone() const override;
    bool IsInitialized(size_t binding_index, uint32_t element) const override;
    uint32_t ResourceCount() const;
    D3D12DescriptorPool::Lease *ResourceDescriptors() const;
    D3D12DescriptorPool::Lease *SamplerDescriptors() const;

   private:
    D3D12Device *device_;
    bool WriteDescriptors(size_t binding_index,
                          uint32_t first_element,
                          const std::vector<ITopLevelAccelerationStructure *> &values) override;
    Version(D3D12Device *device, const BindingLayoutDesc &layout);
    bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<BufferView> &values) override;
    bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<IImage *> &values) override;
    bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<ISampler *> &values) override;
    std::vector<uint32_t> offsets_;
    std::vector<std::vector<bool>> initialized_;
    ObjectPtr<D3D12DescriptorPool::Lease> resources_, samplers_;
  };

 private:
  D3D12BindingGroup(D3D12Device *device, ObjectPtr<Version> version);
};
}  // namespace grassland::rhi::backend::d3d12

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::D3D12BindingGroup::Version)

inline bool grassland::rhi::backend::d3d12::D3D12BindingGroup::Version::QueryInterface(InterfaceID id,
                                                                                       void **output) noexcept {
  if (!output)
    return false;
  if (auto result = MatchInterface<Version>(id, this)) {
    *output = result;
    AddRef();
    return true;
  }
  return BindingGroup::Version::QueryInterface(id, output);
}
