#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendCore)
#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/rhi_core.h"

namespace grassland::rhi::backend {

class IBackendCore : public rhi::ICore {
 public:
  rhi::IDevice *CreateDeviceByAutoSelection(DeviceFeatureRequirement requirement = {}) override = 0;
  rhi::IDevice *CommonCreateDeviceByAutoSelection(DeviceFeatureRequirement requirement = {});
};

IBackendCore *CreateCore(BackendAPI api, const CoreDesc &desc);

// Returns a stable vector index, or -1 when no device satisfies the filters.
int32_t SelectPhysicalDevice(const std::vector<PhysicalDevice> &devices, DeviceFeatureRequirement requirement);

}  // namespace grassland::rhi::backend
