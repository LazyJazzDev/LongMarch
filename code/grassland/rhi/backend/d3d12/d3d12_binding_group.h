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
    ObjectPtr<BindingGroup::Version> Clone() const override;
    bool IsInitialized(size_t binding_index, uint32_t element) const override;
    uint32_t ResourceCount() const;
    // Borrowed CPU-only descriptor leases. Structured views depend on the Program's
    // reflected element strides; variants are cached by stride, not Program identity.
    D3D12DescriptorPool::Lease *ResourceDescriptors(const D3D12Program *program, uint32_t group);
    D3D12DescriptorPool::Lease *SamplerDescriptors() const;

   private:
    Version(D3D12Device *device, const BindingLayoutDesc &layout);
    bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<BufferView> &values) override;
    bool WriteDescriptors(size_t binding_index, uint32_t first_element, const std::vector<IImage *> &values) override;
    bool WriteDescriptors(size_t binding_index,
                          uint32_t first_element,
                          const std::vector<SamplerDesc> &values) override;
    std::vector<uint32_t> offsets_;
    std::vector<std::vector<bool>> initialized_;
    // Only structured buffers need byte ranges until Program reflection supplies stride.
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> structured_ranges_;
    ObjectPtr<D3D12DescriptorPool::Lease> resources_, samplers_;
    std::vector<std::pair<std::vector<uint32_t>, ObjectPtr<D3D12DescriptorPool::Lease>>> resource_variants_;
  };

 private:
  D3D12BindingGroup(D3D12Device *device, ObjectPtr<Version> version);
};
}  // namespace grassland::rhi::backend::d3d12
