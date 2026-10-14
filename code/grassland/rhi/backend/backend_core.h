#pragma once
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/rhi_core.h"

namespace grassland::rhi::backend {

class ICore : public IObject {
 public:
  virtual std::vector<PhysicalDevice> GetPhysicalDevices() const = 0;
  virtual std::unique_ptr<IDevice> CreateDevice(int32_t device_idx, const ObjectPtr<ICore> &owner) const = 0;
};

std::unique_ptr<ICore> CreateCore(BackendAPI api);

// Returns a stable vector index, or -1 when no device satisfies the filters.
int32_t SelectPhysicalDevice(const std::vector<PhysicalDevice> &devices, DeviceFeatureRequirement requirement);

}  // namespace grassland::rhi::backend
