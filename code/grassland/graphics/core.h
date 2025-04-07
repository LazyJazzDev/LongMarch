#pragma once
#ifdef WIN32
#include <d3dcommon.h>
#include <wrl/client.h>
#endif

#include "buffer.h"
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {

class Core {
 public:
  struct Settings {
    int frames_in_flight{2};
    bool enable_debug{kEnableDebug};
  };

  Core(const Settings &settings);

  virtual ~Core() = default;

  virtual BackendAPI API() const = 0;

  virtual int CreateBuffer(size_t size, BufferType type, double_ptr<Buffer> pp_buffer) = 0;

  virtual int CreateImage(int width, int height, ImageFormat format, double_ptr<Image> pp_image) = 0;

  virtual int CreateSampler(const SamplerInfo &info, double_ptr<Sampler> pp_sampler) = 0;

  int CreateWindowObject(int width, int height, const std::string &title, double_ptr<Window> pp_window);
  virtual int CreateWindowObject(int width,
                                 int height,
                                 const std::string &title,
                                 bool fullscreen,
                                 bool resizable,
                                 double_ptr<Window> pp_window) = 0;

  virtual int CreateShader(const std::string &source_code,
                           const std::string &entry_point,
                           const std::string &target,
                           double_ptr<Shader> pp_shader) = 0;

  virtual int CreateProgram(const std::vector<ImageFormat> &color_formats,
                            ImageFormat depth_format,
                            double_ptr<Program> pp_program) = 0;

  virtual int CreateCommandContext(double_ptr<CommandContext> pp_command_context) = 0;

  virtual int CreateBottomLevelAccelerationStructure(Buffer *vertex_buffer,
                                                     Buffer *index_buffer,
                                                     uint32_t stride,
                                                     double_ptr<AccelerationStructure> pp_blas) = 0;

  virtual int CreateTopLevelAccelerationStructure(const std::vector<RayTracingInstance> &instances,
                                                  double_ptr<AccelerationStructure> pp_tlas) = 0;

  virtual int CreateTopLevelAccelerationStructure(
      const std::vector<std::pair<AccelerationStructure *, glm::mat4>> &objects,
      double_ptr<AccelerationStructure> pp_tlas);

  virtual int CreateRayTracingProgram(Shader *raygen_shader,
                                      Shader *miss_shader,
                                      Shader *closest_shader,
                                      double_ptr<RayTracingProgram> pp_program) = 0;

  virtual int SubmitCommandContext(CommandContext *p_command_context) = 0;

  virtual int GetPhysicalDeviceProperties(PhysicalDeviceProperties *p_physical_device_properties = nullptr) = 0;

  virtual int InitializeLogicalDevice(int device_index) = 0;

  virtual void WaitGPU() = 0;

  int InitializeLogicalDeviceAutoSelect(bool require_ray_tracing);

  int FramesInFlight() const;

  virtual uint32_t CurrentFrame() const = 0;

  bool DebugEnabled() const;

  bool DeviceRayTracingSupport() const;

  std::string DeviceName() const;

  static void PyBind(pybind11::module &m);

 private:
  Settings settings_;

 protected:
  std::string device_name_{};
  bool ray_tracing_support_{false};
};

int CreateCore(BackendAPI api, const Core::Settings &settings, double_ptr<Core> pp_core);

}  // namespace grassland::graphics
