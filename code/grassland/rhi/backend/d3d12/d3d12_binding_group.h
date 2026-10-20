#pragma once
#include "grassland/rhi/backend/backend_binding_group.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Device;

class D3D12BindingGroup final : public BindingGroup {
 public:
  static D3D12BindingGroup *Create(D3D12Device *device, const BindingLayoutDesc &layout);

 private:
  class Version final : public BindingGroup::Version {
   public:
    Version(D3D12Device *device, const BindingLayoutDesc &layout);

   private:
    ObjectPtr<BindingGroup::Version> CreateEmpty() const override;
    bool WriteDescriptors(size_t offset, const std::vector<Resource> &resources) override;
  };

  D3D12BindingGroup(D3D12Device *device, ObjectPtr<Version> version);
};
}  // namespace grassland::rhi::backend::d3d12
