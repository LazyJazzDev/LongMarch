#pragma once
#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Device;
class D3D12Program;

class D3D12BindingGroup final : public BindingGroup {
 public:
  static D3D12BindingGroup *Create(D3D12Device *device, const BindingLayoutDesc &layout);

  class Version final : public BindingGroup::Version {
   public:
    static Version *Create(D3D12Device *device, const BindingLayoutDesc &layout);
    uint32_t ResourceCount() const;
    // Borrowed CPU-only descriptor leases. Structured views depend on the Program's
    // reflected element strides; variants are cached by stride, not Program identity.
    D3D12DescriptorPool::Lease *ResourceDescriptors(const D3D12Program *program, uint32_t group);
    D3D12DescriptorPool::Lease *SamplerDescriptors() const;

   private:
    Version(D3D12Device *device, const BindingLayoutDesc &layout);
    ObjectPtr<BindingGroup::Version> CreateEmpty() const override;
    bool WriteDescriptors(size_t offset, const BindingDesc &binding, const std::vector<BufferView> &values) override;
    bool WriteDescriptors(size_t offset, const BindingDesc &binding, const std::vector<IImage *> &values) override;
    bool WriteDescriptors(size_t offset, const BindingDesc &binding, const std::vector<SamplerDesc> &values) override;
    std::vector<uint32_t> offsets_;
    ObjectPtr<D3D12DescriptorPool::Lease> resources_, samplers_;
    std::vector<std::pair<std::vector<uint32_t>, ObjectPtr<D3D12DescriptorPool::Lease>>> resource_variants_;
  };

 private:
  D3D12BindingGroup(D3D12Device *device, ObjectPtr<Version> version);
};
}  // namespace grassland::rhi::backend::d3d12
