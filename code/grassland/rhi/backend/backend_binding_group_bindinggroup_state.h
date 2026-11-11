#pragma once
#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
struct BindingGroupState final {
  BindingGroupState();
  ~BindingGroupState();
  IBackendDevice *device_;
  BindingLayoutDesc layout_;
  ObjectPtr<IBackendBindingVersion> version_;
};
}  // namespace grassland::rhi::backend
