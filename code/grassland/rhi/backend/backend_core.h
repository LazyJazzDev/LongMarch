#pragma once
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/rhi_core.h"

namespace grassland::rhi::backend {

class Core : public Object<rhi::ICore> {
 public:
  rhi::IDevice *CreateDeviceByAutoSelection(DeviceFeatureRequirement requirement = {}) override;
};

std::unique_ptr<Core> CreateCore(BackendAPI api);

// Returns a stable vector index, or -1 when no device satisfies the filters.
int32_t SelectPhysicalDevice(const std::vector<PhysicalDevice> &devices, DeviceFeatureRequirement requirement);

}  // namespace grassland::rhi::backend
