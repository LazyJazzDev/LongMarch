#include "grassland/graphics/core.h"

#include "grassland/graphics/acceleration_structure.h"
#include "grassland/graphics/backend/backend.h"

namespace grassland::graphics {

void Core::Settings::PybindClassRegistration(py::classh<Settings> &c) {
  c.def(py::init<int, bool>(), py::arg("frames_in_flight") = 2, py::arg("enable_debug") = kEnableDebug);
  c.def("__repr__", [](const Settings &settings) {
    return py::str("Core.Settings(frames_in_flight={}, enable_debug={})")
        .format(settings.frames_in_flight, settings.enable_debug);
  });
  c.def_readwrite("frames_in_flight", &Settings::frames_in_flight, "Number of frames buffer");
  c.def_readwrite("enable_debug", &Settings::enable_debug, "Enable debug mode, which may have performance impact");
}

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

void Core::PybindClassRegistration(py::classh<Core> &c) {
  c.doc() = "Core class for resource managements";
  c.def(py::init([](BackendAPI api, const Settings &settings) {
          std::shared_ptr<Core> core_;
          CreateCore(api, settings, &core_);
          return core_;
        }),
        py::arg_v("api", BACKEND_API_DEFAULT, "BackendAPI.BACKEND_API_DEFAULT"), py::arg("settings") = Settings{});
  c.def("__repr__", [](Core *core) {
    if (core->device_name_.empty()) {
      return py::str("Core(<Not Initialized>)");
    } else {
      return py::str("Core(api={}, device='{}')").format(BackendAPIString(core->API()), core->device_name_);
    }
  });
  c.def("init", &Core::InitializeLogicalDevice, py::arg("device_index"), "Initialize logical device by device index");
  c.def("init_auto", &Core::InitializeLogicalDeviceAutoSelect, py::arg("require_ray_tracing") = false,
        "Auto select and initialize logical device, will select device with raytracing support if required");
  c.def("api", &Core::API, "Get the backend API");
  c.def("device_name", &Core::DeviceName, "Get the device name");
  c.def("ray_tracing_support", &Core::DeviceRayTracingSupport, "Check if the device supports ray tracing");
  c.def("debug_enabled", &Core::DebugEnabled, "Check if debug mode is enabled");
  c.def("frames_in_flight", &Core::FramesInFlight, "Get number of frames in flight");
  c.def("current_frame", &Core::CurrentFrame, "Get current frame index");
  c.def("wave_size", &Core::WaveSize, "Get the wave size of the device");
  c.def("wait_gpu", &Core::WaitGPU, "Wait for the GPU to finish all operations");

  c.def(
      "create_window",
      [](Core *core, int width, int height, const std::string &title, bool fullscreen, bool resizable) {
        std::shared_ptr<Window> window_;
        core->CreateWindowObject(width, height, title, fullscreen, resizable, &window_);
        return window_;
      },
      py::arg("width"), py::arg("height"), py::arg("title"), py::arg("fullscreen") = false,
      py::arg("resizable") = false, "Create a window", py::keep_alive<0, 1>{});

  c.def(
      "create_buffer",
      [](Core *core, size_t buffer, BufferType type) {
        std::shared_ptr<Buffer> buffer_;
        core->CreateBuffer(buffer, type, &buffer_);
        return buffer_;
      },
      py::arg("size"), py::arg_v("type", BUFFER_TYPE_STATIC, "type"), "Create a buffer", py::keep_alive<0, 1>{});

  c.def(
      "create_image",
      [](Core *core, int width, int height, ImageFormat format) {
        std::shared_ptr<Image> image_;
        core->CreateImage(width, height, format, &image_);
        return image_;
      },
      py::arg("width"), py::arg("height"), py::arg_v("format", IMAGE_FORMAT_R8G8B8A8_UNORM, "format"),
      "Create an image", py::keep_alive<0, 1>{});

  c.def(
      "create_sampler",
      [](Core *core, const SamplerInfo &info) {
        std::shared_ptr<Sampler> sampler_;
        core->CreateSampler(info, &sampler_);
        return sampler_;
      },
      py::arg_v("info", SamplerInfo{}, "SamplerInfo()"), "Create a sampler", py::keep_alive<0, 1>{});

  c.def(
      "create_shader",
      [](Core *core, const std::string &source_code, const std::string &entry_point, const std::string &target,
         const std::vector<std::string> &args) {
        std::shared_ptr<Shader> shader_;
        VirtualFileSystem vfs;
        vfs.WriteFile("main.hlsl", source_code);
        core->CreateShader(vfs, "main.hlsl", entry_point, target, args, &shader_);
        return shader_;
      },
      py::arg("source_code"), py::arg("entry_point"), py::arg("target"), py::arg("args") = std::vector<std::string>{},
      "Create shader from source code", py::keep_alive<0, 1>{});

  c.def(
      "create_program",
      [](Core *core, const std::vector<ImageFormat> &color_formats, ImageFormat depth_format) {
        std::shared_ptr<Program> program_;
        core->CreateProgram(color_formats, depth_format, &program_);
        return program_;
      },
      py::arg("color_formats"), py::arg_v("depth_format", IMAGE_FORMAT_UNDEFINED, "depth_format"),
      "Create a graphics program", py::keep_alive<0, 1>{});

  c.def(
      "create_compute_program",
      [](Core *core, Shader *compute_shader) {
        std::shared_ptr<ComputeProgram> program_;
        core->CreateComputeProgram(compute_shader, &program_);
        return program_;
      },
      py::arg("compute_shader"), "Create a compute program", py::keep_alive<0, 1>{}, py::keep_alive<0, 2>{});

  c.def(
      "create_raytracing_program",
      [](Core *core) {
        std::shared_ptr<RayTracingProgram> program_;
        core->CreateRayTracingProgram(&program_);
        return program_;
      },
      "Create a ray tracing program", py::keep_alive<0, 1>{});

  c.def(
      "create_raytracing_program",
      [](Core *core, Shader *raygen_shader, Shader *miss_shader, Shader *closest_hit_shader) {
        std::shared_ptr<RayTracingProgram> program_;
        core->CreateRayTracingProgram(raygen_shader, miss_shader, closest_hit_shader, &program_);
        return program_;
      },
      py::arg("raygen_shader"), py::arg("miss_shader"), py::arg("closest_hit_shader"),
      "Create a ray tracing program with shaders", py::keep_alive<0, 1>{}, py::keep_alive<0, 2>{},
      py::keep_alive<0, 3>{}, py::keep_alive<0, 4>{});

  c.def(
      "create_blas",
      [](Core *core, BufferRange aabb_buffer, uint32_t stride, uint32_t num_aabb, RayTracingGeometryFlag flags) {
        std::shared_ptr<AccelerationStructure> blas_;
        core->CreateBottomLevelAccelerationStructure(aabb_buffer, stride, num_aabb, flags, &blas_);
        return blas_;
      },
      py::arg("aabb_buffer"), py::arg("stride"), py::arg("num_aabb"), py::arg("flags"),
      "Create bottom-level acceleration structure from AABB buffer", py::keep_alive<0, 1>{});

  c.def(
      "create_blas",
      [](Core *core, BufferRange vertex_buffer, BufferRange index_buffer, uint32_t num_vertex, uint32_t stride,
         uint32_t num_primitive, RayTracingGeometryFlag flags) {
        std::shared_ptr<AccelerationStructure> blas_;
        core->CreateBottomLevelAccelerationStructure(vertex_buffer, index_buffer, num_vertex, stride, num_primitive,
                                                     flags, &blas_);
        return blas_;
      },
      py::arg("vertex_buffer"), py::arg("index_buffer"), py::arg("num_vertex"), py::arg("stride"),
      py::arg("num_primitive"), py::arg("flags"),
      "Create bottom-level acceleration structure from vertex and index buffers", py::keep_alive<0, 1>{});

  c.def(
      "create_blas",
      [](Core *core, Buffer *vertex_buffer, Buffer *index_buffer, uint32_t stride) {
        std::shared_ptr<AccelerationStructure> blas_;
        core->CreateBottomLevelAccelerationStructure(vertex_buffer, index_buffer, stride, &blas_);
        return blas_;
      },
      py::arg("vertex_buffer"), py::arg("index_buffer"), py::arg("stride"),
      "Create bottom-level acceleration structure from vertex and index buffer objects", py::keep_alive<0, 1>{},
      py::keep_alive<0, 2>{}, py::keep_alive<0, 3>{});

  c.def(
      "create_tlas",
      [](Core *core, const std::vector<RayTracingInstance> &instances) {
        std::shared_ptr<AccelerationStructure> tlas_;
        core->CreateTopLevelAccelerationStructure(instances, &tlas_);
        return tlas_;
      },
      py::arg("instances"), "Create top-level acceleration structure from ray tracing instances",
      py::keep_alive<0, 1>{});

  c.def(
      "create_tlas",
      [](Core *core, py::list objects_list) {
        std::vector<std::pair<AccelerationStructure *, glm::mat4>> objects;
        for (auto item : objects_list) {
          py::tuple pair = item.cast<py::tuple>();
          if (pair.size() != 2) {
            throw std::runtime_error("Each object must be a pair of (AccelerationStructure, transform)");
          }
          AccelerationStructure *as = pair[0].cast<AccelerationStructure *>();
          py::list transform_list = pair[1].cast<py::list>();

          if (transform_list.size() != 4) {
            throw std::runtime_error("Transform matrix must have 4 rows");
          }
          glm::mat4 transform;
          for (int i = 0; i < 4; i++) {
            py::list row = transform_list[i].cast<py::list>();
            if (row.size() != 4) {
              throw std::runtime_error("Transform matrix rows must have 4 columns");
            }
            for (int j = 0; j < 4; j++) {
              transform[i][j] = row[j].cast<float>();
            }
          }
          objects.emplace_back(as, transform);
        }
        std::shared_ptr<AccelerationStructure> tlas_;
        core->CreateTopLevelAccelerationStructure(objects, &tlas_);
        return tlas_;
      },
      py::arg("objects"), "Create top-level acceleration structure from acceleration structure and transform pairs",
      py::keep_alive<0, 1>{});

  c.def(
      "create_command_context",
      [](Core *core) {
        std::shared_ptr<CommandContext> command_context_;
        core->CreateCommandContext(&command_context_);
        return command_context_;
      },
      "Create a command context", py::keep_alive<0, 1>{});

  c.def("submit_command_context", &Core::SubmitCommandContext, py::arg("command_context"),
        "Submit a command context to the GPU queue");

#if defined(LONGMARCH_CUDA_RUNTIME)
  c.def("init_cuda", &Core::InitializeLogicalDeviceByCUDADeviceID, py::arg("cuda_device_id"),
        "Initialize logical device by CUDA device index");
#endif
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
