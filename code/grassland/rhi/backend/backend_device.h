#pragma once
#include <utility>

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {

class IDevice : public IObject {
 public:
  IDevice(PhysicalDevice info, const ObjectPtr<ICore> &core);
  ~IDevice() override;

  const PhysicalDevice &GetPhysicalDevice() const noexcept {
    return info_;
  }

 private:
  // Backends destroy their native device before this reference releases the Core.
  ObjectPtr<ICore> core_;
  PhysicalDevice info_;
};

}  // namespace grassland::rhi::backend
