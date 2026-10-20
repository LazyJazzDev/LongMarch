#pragma once
#include "grassland/rhi/rhi_binding_group.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
class BindingGroup : public ObjectContainer {
 public:
  BindingGroup() noexcept = default;

  BindingGroup(rhi::IBindingGroup *object) noexcept : ObjectContainer(object) {
  }

  rhi::IBindingGroup *Get() const noexcept {
    return static_cast<rhi::IBindingGroup *>(GetObject());
  }
};
}  // namespace grassland::rhix
