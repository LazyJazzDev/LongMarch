#include "grassland/graphics/core.h"

#include "grassland/graphics/acceleration_structure.h"
#include "grassland/graphics/backend/backend.h"

namespace CD::graphics {

Core::Core(const Settings &settings) : settings_(settings) {
}

int Core::CreateWindowObject(int width, int height, const std::string &title, double_ptr<Window> pp_window) {
  return CreateWindowObject(width, height, title, false, false, pp_window);
}

int Core::CreateTopLevelAccelerationStructure(const std::vector<std::pair<AccelerationStructure *, glm::mat4>> &objects,
                                              double_ptr<AccelerationStructure> pp_tlas) {
  std::vector<RayTracingInstance> instances(objects.size());
  for (int i = 0; i < objects.size(); i++) {
    auto &object = objects[i];
    instances[i] = object.first->MakeInstance(object.second, i);
  }
  return CreateTopLevelAccelerationStructure(instances, pp_tlas);
}

int Core::CreateRayTracingProgram(Shader *raygen_shader,
                                  Shader *miss_shader,
                                  Shader *closest_shader,
                                  double_ptr<RayTracingProgram> pp_program) {
  int ret = CreateRayTracingProgram(pp_program);
  if (!ret) {
    pp_program->AddRayGenShader(raygen_shader);
    pp_program->AddMissShader(miss_shader);
    pp_program->AddHitGroup(closest_shader);
  }
  return ret;
}

int Core::InitializeLogicalDeviceAutoSelect(bool require_ray_tracing) {
  auto num_device = GetPhysicalDeviceProperties();
  std::vector<PhysicalDeviceProperties> device_properties(num_device);
  GetPhysicalDeviceProperties(device_properties.data());
  int device_index = -1;
  uint64_t max_score = 0;
  for (int i = 0; i < device_properties.size(); i++) {
    // LogInfo("Device: {} score={}", device_properties[i].name, device_properties[i].score);
    if (require_ray_tracing && !device_properties[i].ray_tracing_support) {
      continue;
    }
    if (device_index == -1 || max_score < device_properties[i].score) {
      device_index = i;
      max_score = device_properties[i].score;
    }
  }

  if (device_index == -1) {
    LogError("[graphics] Required device not found. Require ray tracing: {}", require_ray_tracing);
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

#if defined(LONGMARCH_CUDA_RUNTIME)
int Core::CUDADeviceIndex() const {
  return cuda_device_;
}

int Core::InitializeLogicalDeviceByCUDADeviceID(int cuda_device_id) {
  auto num_device = GetPhysicalDeviceProperties();
  std::vector<PhysicalDeviceProperties> device_properties(num_device);
  GetPhysicalDeviceProperties(device_properties.data());
  int device_index = -1;
  for (int i = 0; i < device_properties.size(); i++) {
    if (device_properties[i].cuda_device_index == cuda_device_id) {
      device_index = i;
    }
  }

  if (device_index == -1) {
    LogError("[graphics] Required device not found. Required CUDA device index: {}", cuda_device_id);
  }

  return InitializeLogicalDevice(device_index);
}
#endif

void Core::PyBind(pybind11::module &m) {
  pybind11::class_<Core::Settings> core_settings_class(m, "CoreSettings");
  core_settings_class.def(pybind11::init<int, bool>(), pybind11::arg("frames_in_flight") = 2,
                          pybind11::arg("enable_debug") = kEnableDebug);
  core_settings_class.def_readwrite("frames_in_flight", &Settings::frames_in_flight);
  core_settings_class.def_readwrite("enable_debug", &Settings::enable_debug);
  core_settings_class.def("__repr__", [](const Core::Settings &settings) {
    return pybind11::str("CoreSettings(frames_in_flight={}, enable_debug={})")
        .format(settings.frames_in_flight, settings.enable_debug);
  });

  // m.def(
  //     "create_core",
  //     [](BackendAPI api, const Core::Settings &settings, bool ray_tracing, int device_index) {
  //       std::shared_ptr<Core> core;
  //       CreateCore(api, settings, &core);
  //       if (device_index == -1) {
  //         core->InitializeLogicalDeviceAutoSelect(ray_tracing);
  //       } else {
  //         core->InitializeLogicalDevice(device_index);
  //       }
  //       return core;
  //     },
  //     pybind11::arg("api") = BACKEND_API_VULKAN, pybind11::arg("settings") = Core::Settings{},
  //     pybind11::arg("ray_tracing") = false, pybind11::arg("device_index") = -1);

  pybind11::class_<Core, std::shared_ptr<Core>> core_class(m, "Core");
  core_class.attr("Settings") = core_settings_class;
  core_class.def("__repr__", [](const Core &core) {
    return pybind11::str("Core(API={}, DeviceName=\"{}\", RayTracingSupport={}, FramesInFlight={}, DebugEnabled={})")
        .format(core.API(), core.DeviceName(), core.DeviceRayTracingSupport(), core.FramesInFlight(),
                core.DebugEnabled());
  });
  core_class.def(pybind11::init([](BackendAPI api, const Core::Settings &settings, bool ray_tracing, int device_index) {
                   std::shared_ptr<Core> core;
                   CreateCore(api, settings, &core);
                   if (device_index == -1) {
                     core->InitializeLogicalDeviceAutoSelect(ray_tracing);
                   } else {
                     core->InitializeLogicalDevice(device_index);
                   }
                   return core;
                 }),
                 pybind11::arg("api") = BACKEND_API_VULKAN, pybind11::arg("settings") = Core::Settings{},
                 pybind11::arg("ray_tracing") = false, pybind11::arg("device_index") = -1);
  core_class.def("API", &Core::API);
  core_class.def("device_name", &Core::DeviceName);
  core_class.def("ray_tracing", &Core::DeviceRayTracingSupport);
  core_class.def("frames_in_flight", &Core::FramesInFlight);
  core_class.def("debug_enabled", &Core::DebugEnabled);
  core_class.def("wait_gpu", &Core::WaitGPU);
  core_class.def(
      "create_window",
      [](std::shared_ptr<Core> core, int width, int height, const std::string &title, bool fullscreen, bool resizable) {
        std::shared_ptr<Window> window;
        core->CreateWindowObject(width, height, title, fullscreen, resizable, &window);
        return window;
      },
      pybind11::arg("width") = 800, pybind11::arg("height") = 600, pybind11::arg("title") = "Grassland",
      pybind11::arg("fullscreen") = false, pybind11::arg("resizable") = false, pybind11::keep_alive<0, 1>{});

  core_class.def(
      "create_image",
      [](std::shared_ptr<Core> core, int width, int height, ImageFormat format) {
        std::shared_ptr<Image> image;
        core->CreateImage(width, height, format, &image);
        return image;
      },
      pybind11::arg("width"), pybind11::arg("height"), pybind11::arg("format") = IMAGE_FORMAT_R8G8B8A8_UNORM,
      pybind11::keep_alive<0, 1>{});

  core_class.def(
      "create_command_context",
      [](std::shared_ptr<Core> core) {
        std::shared_ptr<CommandContext> context;
        core->CreateCommandContext(&context);
        return context;
      },
      pybind11::keep_alive<0, 1>{});
  core_class.def("submit_command_context", &Core::SubmitCommandContext);
}

int CreateCore(BackendAPI api, const Core::Settings &settings, double_ptr<Core> pp_core) {
  switch (api) {
#ifdef _WIN64
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

}  // namespace CD::graphics
