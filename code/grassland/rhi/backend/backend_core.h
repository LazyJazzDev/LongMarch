#pragma once
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/rhi_core.h"

namespace grassland::rhi::backend {

class Core : public Object<rhi::ICore> {
 public:
  rhi::IDevice *CreateDevice(int32_t device_idx) override;
  rhi::IDevice *CreateDeviceByAutoSelection(DeviceFeatureRequirement requirement = {}) override;
  virtual std::vector<PhysicalDevice> GetPhysicalDevices() const = 0;
  virtual std::unique_ptr<Device> CreateNativeDevice(int32_t device_idx, Core *owner) const = 0;
};

std::unique_ptr<Core> CreateCore(BackendAPI api);

// Returns a stable vector index, or -1 when no device satisfies the filters.
int32_t SelectPhysicalDevice(const std::vector<PhysicalDevice> &devices, DeviceFeatureRequirement requirement);

}  // namespace grassland::rhi::backend
