#pragma once
#include "grassland/graphics/core.h"
#include "grassland/graphics/graphics.h"
#include "sparkium/scene/render_types.h"

namespace sparkium {

struct BackendSelection {
  RenderBackend backend{RenderBackend::Graphics};
  grassland::graphics::BackendAPI graphics_api{grassland::graphics::BACKEND_API_DEFAULT};
  BackendSelection() = default;

  BackendSelection(RenderBackend kind, grassland::graphics::BackendAPI api = grassland::graphics::BACKEND_API_DEFAULT)
      : backend(kind),
        graphics_api(api) {
  }
};

const char *BackendName(RenderBackend backend);
const char *BackendName(BackendSelection selection);
bool SupportBackend(BackendSelection selection);
grassland::graphics::BackendAPI ToGraphicsBackend(BackendSelection selection);

namespace backend {
using namespace grassland;
using namespace grassland::graphics;

// Rendering resources use the existing abstract resource contracts. Native
// implementations and scheduling are owned by Sparkium, never graphics::Core.
class Device {
 public:
  struct Settings {
    int frames_in_flight{2};
    bool enable_debug{kEnableDebug};
  };

  explicit Device(const Settings &settings) : settings_(settings) {
  }

  virtual ~Device() = default;
  virtual RenderBackend API() const = 0;

  // Null for CPU/CUDA. Presentation and graphics-only operations use this bridge.
  virtual graphics::Core *GraphicsCore() const {
    return nullptr;
  }

  virtual int CreateBuffer(size_t size, BufferType type, double_ptr<Buffer> pp_buffer) = 0;

  virtual int CreateImage(int width, int height, ImageFormat format, double_ptr<Image> pp_image) = 0;

  int LoadImage(const std::string &path, double_ptr<Image> image);

  virtual int CreateSampler(const SamplerInfo &info, double_ptr<Sampler> pp_sampler) = 0;

  virtual int CreateShader(const std::string &source_code,
                           const std::string &entry_point,
                           const std::string &target,
                           double_ptr<Shader> pp_shader) = 0;

  virtual int CreateShader(const VirtualFileSystem &vfs,
                           const std::string &source_file,
                           const std::string &entry_point,
                           const std::string &target,
                           double_ptr<Shader> pp_shader) = 0;

  virtual int CreateShader(const VirtualFileSystem &vfs,
                           const std::string &source_file,
                           const std::string &entry_point,
                           const std::string &target,
                           const std::vector<std::string> &args,
                           double_ptr<Shader> pp_shader) = 0;

  virtual int CreateProgram(const std::vector<ImageFormat> &color_formats,
                            ImageFormat depth_format,
                            double_ptr<Program> pp_program) = 0;

  virtual int CreateComputeProgram(Shader *compute_shader, double_ptr<ComputeProgram> pp_program) = 0;

  virtual int CreateCommandContext(double_ptr<CommandContext> pp_command_context) = 0;

  virtual int CreateBottomLevelAccelerationStructure(BufferRange aabb_buffer,
                                                     uint32_t stride,
                                                     uint32_t num_aabb,
                                                     RayTracingGeometryFlag flags,
                                                     double_ptr<AccelerationStructure> pp_blas) = 0;

  virtual int CreateBottomLevelAccelerationStructure(BufferRange vertex_buffer,
                                                     BufferRange index_buffer,
                                                     uint32_t num_vertex,
                                                     uint32_t stride,
                                                     uint32_t num_primitive,
                                                     RayTracingGeometryFlag flags,
                                                     double_ptr<AccelerationStructure> pp_blas) = 0;

  virtual int CreateBottomLevelAccelerationStructure(Buffer *vertex_buffer,
                                                     Buffer *index_buffer,
                                                     uint32_t stride,
                                                     double_ptr<AccelerationStructure> pp_blas) = 0;

  virtual int CreateTopLevelAccelerationStructure(const std::vector<RayTracingInstance> &instances,
                                                  double_ptr<AccelerationStructure> pp_tlas) = 0;

  virtual int CreateRayTracingProgram(double_ptr<RayTracingProgram> pp_program) = 0;

  virtual int SubmitCommandContext(CommandContext *p_command_context) = 0;

  virtual int GetPhysicalDeviceProperties(PhysicalDeviceProperties *p_physical_device_properties = nullptr) = 0;

  virtual int InitializeLogicalDevice(int device_index) = 0;

  virtual void WaitGPU() = 0;

  virtual uint32_t WaveSize() const = 0;

  int InitializeLogicalDeviceAutoSelect(bool require_ray_tracing);

  virtual int FramesInFlight() const {
    return settings_.frames_in_flight;
  }

  virtual uint32_t CurrentFrame() const = 0;

  bool DebugEnabled() const {
    return settings_.enable_debug;
  }

  virtual bool DeviceRayTracingSupport() const {
    return ray_tracing_support_;
  }

  virtual bool DeviceRayQuerySupport() const {
    return false;
  }

  virtual std::string DeviceName() const {
    return device_name_;
  }

 protected:
  Settings settings_;
  std::string device_name_;
  bool ray_tracing_support_{};
};
}  // namespace backend

int CreateDevice(BackendSelection selection,
                 const backend::Device::Settings &settings,
                 grassland::double_ptr<backend::Device> device);
}  // namespace sparkium
