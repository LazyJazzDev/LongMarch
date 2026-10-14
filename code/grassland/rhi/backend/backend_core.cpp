#include "grassland/rhi/backend/backend_core.h"

#include <algorithm>
#include <limits>

#if defined(LONGMARCH_D3D12_ENABLED)
#include "grassland/rhi/backend/d3d12/d3d12_core.h"
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
#include "grassland/rhi/backend/vulkan/vulkan_core.h"
#endif
#if defined(LONGMARCH_METAL_ENABLED)
#include "grassland/rhi/backend/metal/metal_core.h"
#endif

namespace grassland::rhi::backend {

rhi::IDevice *Core::CreateDevice(int32_t device_idx) {
  try {
    const auto devices = GetPhysicalDevices();
    if (device_idx < 0 || static_cast<size_t>(device_idx) >= devices.size()) {
      ReportError(-1, "Physical device ID is out of range.");
      return nullptr;
    }
    return CreateNativeDevice(device_idx, this).release();
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}

rhi::IDevice *Core::CreateDeviceByAutoSelection(DeviceFeatureRequirement requirement) {
  try {
    const auto id = SelectPhysicalDevice(GetPhysicalDevices(), requirement);
    if (id < 0) {
      ReportError(-1, "No physical device satisfies the requested RHI features.");
      return nullptr;
    }
    return CreateDevice(id);
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}

std::unique_ptr<Core> CreateCore(BackendAPI api) {
  switch (api) {
#if defined(LONGMARCH_D3D12_ENABLED)
    case BACKEND_API_D3D12:
      return d3d12::D3D12Core::CreateD3D12Core();
#endif
#if defined(LONGMARCH_VULKAN_ENABLED)
    case BACKEND_API_VULKAN:
      return vulkan::VulkanCore::CreateVulkanCore();
#endif
#if defined(LONGMARCH_METAL_ENABLED)
    case BACKEND_API_METAL:
      return metal::MetalCore::CreateMetalCore();
#endif
    default:
      ReportError(-1, "Requested RHI backend is unavailable in this build.");
      return {};
  }
}

int32_t SelectPhysicalDevice(const std::vector<PhysicalDevice> &devices, DeviceFeatureRequirement requirement) {
  int32_t best = -1;
  uint64_t best_score = 0;
  for (size_t i = 0; i < devices.size() && i <= static_cast<size_t>(std::numeric_limits<int32_t>::max()); ++i) {
    const auto &device = devices[i];
    if ((requirement.ray_tracing_pipeline && !device.ray_tracing_pipeline_feature) ||
        (requirement.ray_query && !device.ray_query_feature) ||
        (requirement.geometry_shader && !device.geometry_shader_feature)) {
      continue;
    }
    // General rendering heuristic, not a performance measurement: dedicated > integrated > software/other.
    // Within each class, combine memory (256 MiB per point, capped at 64 GiB) and feature coverage.
    const uint64_t memory =
        device.type == PHYSICAL_DEVICE_TYPE_DEDICATED ? device.dedicated_memory : device.unified_memory;
    const uint64_t score = (device.type == PHYSICAL_DEVICE_TYPE_DEDICATED    ? 1000
                            : device.type == PHYSICAL_DEVICE_TYPE_INTEGRATED ? 500
                                                                             : 0) +
                           std::min<uint64_t>(memory / (256ull * 1024 * 1024), 256) +
                           100 * device.ray_tracing_pipeline_feature + 50 * device.ray_query_feature +
                           10 * device.geometry_shader_feature;
    if (best < 0 || score > best_score) {
      best = static_cast<int32_t>(i);
      best_score = score;
    }
  }
  return best;
}

}  // namespace grassland::rhi::backend
