#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_binding_group.h"

namespace grassland::rhi::backend {
class Device;

class BindingGroup : public Object<IBindingGroup> {
 public:
  BindingGroup(Device *device, const BindingLayoutDesc &layout, const std::vector<BindingEntry> &entries);
  const BindingLayoutDesc &GetLayout() const override;
  const std::vector<BindingEntry> &GetEntries() const override;
  Device *Owner() const;
  static bool Validate(Device *device, const BindingLayoutDesc &layout, const std::vector<BindingEntry> &entries);

 protected:
  Device *device_;
  BindingLayoutDesc layout_;
  std::vector<BindingEntry> entries_;

 private:
  std::vector<ObjectPtr<IObject>> resources_;
};

struct MappedBinding {
  BindingDesc desc;
  uint32_t group{}, shader_register{}, table_offset{};
};

std::vector<MappedBinding> MapBindings(const PipelineLayoutDesc &layout, BackendAPI api);
bool ValidateBindingLayout(const BindingLayoutDesc &layout);
bool SameBindingLayout(const BindingLayoutDesc &a, const BindingLayoutDesc &b);
}  // namespace grassland::rhi::backend
