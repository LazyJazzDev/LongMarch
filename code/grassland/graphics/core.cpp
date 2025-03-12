#include "grassland/graphics/core.h"

#include "../../../demo/d3d12_hello_cube/d3d12app.h"
#include "grassland/graphics/backend/backend.h"

namespace grassland::graphics {

Core::Core(const Settings &settings) : settings_(settings) {
}

int Core::CreateWindowObject(int width, int height, const std::string &title, double_ptr<Window> pp_window) {
  return CreateWindowObject(width, height, title, false, false, pp_window);
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

void Core::PybindModuleRegistration(pybind11::module &m) {
  pybind11::class_<Core::Settings> core_settings_class(m, "CoreSettings");
  core_settings_class.def(pybind11::init<>());
  core_settings_class.def(pybind11::init<int>());
  core_settings_class.def(pybind11::init<int, bool>());
  core_settings_class.def_readwrite("frames_in_flight", &Settings::frames_in_flight);
  core_settings_class.def_readwrite("enable_debug", &Settings::enable_debug);
  core_settings_class.def("__repr__", [](const Core::Settings &settings) {
    return pybind11::str("CoreSettings(frames_in_flight={}, enable_debug={})")
        .format(settings.frames_in_flight, settings.enable_debug);
  });

  m.def(
      "create_core",
      [](BackendAPI api, const Core::Settings &settings, bool ray_tracing, int device_index) {
        std::shared_ptr<Core> core;
        CreateCore(api, settings, &core);
        if (device_index == -1) {
          core->InitializeLogicalDeviceAutoSelect(ray_tracing);
        } else {
          core->InitializeLogicalDevice(device_index);
        }
        return core;
      },
      pybind11::arg("api") = BACKEND_API_VULKAN, pybind11::arg("settings") = Core::Settings{},
      pybind11::arg("ray_tracing") = false, pybind11::arg("device_index") = -1);

  pybind11::class_<Core, std::shared_ptr<Core>> core_class(m, "Core");
  core_class.attr("Settings") = core_settings_class;
  core_class.def("__repr__", [](const Core &core) {
    return pybind11::str("Core(API={}, DeviceName=\"{}\", RayTracingSupport={}, FramesInFlight={}, DebugEnabled={})")
        .format(core.API(), core.DeviceName(), core.DeviceRayTracingSupport(), core.FramesInFlight(),
                core.DebugEnabled());
  });
  core_class.def("API", &Core::API);
  core_class.def("frames_in_flight", &Core::FramesInFlight);
  core_class.def("debug_enabled", &Core::DebugEnabled);
}

int CreateCore(BackendAPI api, const Core::Settings &settings, double_ptr<Core> pp_core) {
  switch (api) {
#if defined(_WIN32)
    case BACKEND_API_D3D12:
      pp_core.construct<backend::D3D12Core>(settings);
      break;
#else
    case BACKEND_API_D3D12:
      LogInfo("D3D12 is not supported on this platform, falling back to Vulkan");
#endif
    case BACKEND_API_VULKAN:
      pp_core.construct<backend::VulkanCore>(settings);
      break;
    default:
      throw std::runtime_error("Unsupported backend API");
      return -1;
  }
  return 0;
}

}  // namespace grassland::graphics
