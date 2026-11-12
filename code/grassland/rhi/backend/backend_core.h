#pragma once
#include <array>
#include <atomic>
#include <mutex>
#include <thread>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_core.h"
#include "grassland/rhi/rhi_device.h"

namespace grassland::rhi::backend {

IDevice *CreateDeviceByAutoSelection(ICore *core, DeviceFeatureRequirement requirement = {});

ICore *CreateCore(BackendAPI api, const CoreDesc &desc);

// Returns a stable vector index, or -1 when no device satisfies the filters.
int32_t SelectPhysicalDevice(const std::vector<PhysicalDevice> &devices, DeviceFeatureRequirement requirement);

}  // namespace grassland::rhi::backend
