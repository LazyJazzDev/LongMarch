#include "grassland/graphics/core.h"

#include "backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/backend.h"

namespace grassland::graphics {

Core::Core(const Settings &settings) : settings_(settings) {
}
int Core::InitializeLogicalDeviceAutoSelect(bool require_ray_tracing) {
  auto num_device = GetPhysicalDeviceProperties();
  std::vector<PhysicalDeviceProperties> device_properties(num_device);
  GetPhysicalDeviceProperties(device_properties.data());
  int device_index = -1;
  uint64_t max_score = 0;
  for (int i = 0; i < device_properties.size(); i++) {
    if (require_ray_tracing && !device_properties[i].ray_tracing_support) {
      continue;
    }
    if (device_index == -1 || max_score < device_properties[i].score) {
      device_index = i;
      max_score = device_properties[i].score;
    }
  }

  return InitializeLogicalDevice(device_index);
}

int Core::FramesInFlight() const {
  return settings_.frames_in_flight;
}

bool Core::DebugEnabled() const {
  return settings_.enable_debug;
}

bool Core::DeviceRayTracingSupport() const {
  return ray_tracing_support_;
}

std::string Core::DeviceName() const {
  return device_name_;
}

int CreateCore(BackendAPI api,
               const Core::Settings &settings,
               double_ptr<Core> pp_core) {
  switch (api) {
    case BACKEND_API_VULKAN:
      pp_core.construct<backend::VulkanCore>(settings);
      break;
    case BACKEND_API_D3D12:
      pp_core.construct<backend::D3D12Core>(settings);
      break;
    default:
      return -1;
  }
  return 0;
}

}  // namespace grassland::graphics
