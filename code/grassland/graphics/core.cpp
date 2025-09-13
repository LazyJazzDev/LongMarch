#include "grassland/graphics/core.h"

#include "grassland/graphics/acceleration_structure.h"
#include "grassland/graphics/backend/backend.h"

namespace grassland::graphics {

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

#if defined(LONGMARCH_D3D12_ENABLED)
#define DEFAULT_API 0
#elif defined(LONGMARCH_VULKAN_ENABLED)
#define DEFAULT_API 1
#endif

void Core::PybindModuleRegistration(py::module_ &m) {
  py::class_<Settings> core_settings(m, "CoreSettings");
  core_settings.def(py::init<int, bool>(), py::arg("frams_in_flight") = 2, py::arg("enable_debug") = kEnableDebug);
  py::class_<Core, py::smart_holder> core(m, "Core");
  core.def(py::init([](BackendAPI api, const Settings &settings) {
    std::shared_ptr<Core> core_;
    CreateCore(api, settings, &core_);
    return core_;
  }));
  core.def(
      "create_shader",
      [](Core *core, const std::string &source_code, const std::string &entry_point, const std::string &target) {
        std::shared_ptr<Shader> shader_;
        core->CreateShader(source_code, entry_point, target, &shader_);
        return shader_;
      },
      py::arg("source_code"), py::arg("entry_point"), py::arg("target"));
}

int CreateCore(BackendAPI api, const Core::Settings &settings, double_ptr<Core> pp_core) {
  switch (api) {
#ifdef LONGMARCH_D3D12_ENABLED
#if DEFAULT_API == 0
    default:
      LogWarning("[graphics] Unsupported backend API <{}>, falling back to D3D12", BackendAPIString(api));
#endif
    case BACKEND_API_D3D12:
      pp_core.construct<backend::D3D12Core>(settings);
      break;
#endif
#ifdef LONGMARCH_VULKAN_ENABLED
#if DEFAULT_API == 1
    default:
      LogWarning("[graphics] Unsupported backend API <{}>, falling back to Vulkan", BackendAPIString(api));
#endif
    case BACKEND_API_VULKAN:
      pp_core.construct<backend::VulkanCore>(settings);
      break;
#endif
#if !defined(DEFAULT_API)
    default:
      LogError("[graphics] No supported graphics API", BackendAPIString(api));
      return -1;
#endif
  }
  return 0;
}

}  // namespace grassland::graphics
