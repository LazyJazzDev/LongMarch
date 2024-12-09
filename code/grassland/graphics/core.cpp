#include "grassland/graphics/core.h"

#include "backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/backend.h"

namespace grassland::graphics {

Core::Core(const Settings &settings) : settings_(settings) {
}

int Core::FramesInFlight() const {
  return settings_.frames_in_flight;
}

bool Core::DebugEnabled() const {
  return settings_.enable_debug;
}

bool Core::RayTracingEnabled() const {
  return settings_.enable_ray_tracing;
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
