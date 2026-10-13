#pragma once
#include <utility>

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {

class IDevice : public IObject {
 public:
  IDevice(PhysicalDevice info, const ObjectRef &core) : core_(core), info_(std::move(info)) {
  }

  const PhysicalDevice &GetPhysicalDevice() const noexcept {
    return info_;
  }

 private:
  // Backends destroy their native device before this reference releases the Core.
  ObjectRef core_;
  PhysicalDevice info_;
};

}  // namespace grassland::rhi::backend
