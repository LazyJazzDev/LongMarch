#pragma once
#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
struct BindingVersionState final {
  BindingVersionState();
  ~BindingVersionState();
  BindingLayoutDesc layout_;
  // Indexed by sorted layout binding, then array element.
  std::vector<std::vector<ObjectPtr<IObject>>> references_;
};
}  // namespace grassland::rhi::backend
