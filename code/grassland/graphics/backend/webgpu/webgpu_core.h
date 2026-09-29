#pragma once
#include "grassland/graphics/backend/webgpu/webgpu_util.h"

namespace grassland::graphics::backend {

// WebGPU in a browser. Adapters and devices are requested asynchronously, so the
// host acquires the device first and provides it with SetDevice before creating
// a core. Browser threads cannot block on the GPU: WaitGPU returns immediately,
// which is safe because WebGPU keeps objects alive while queued work uses them,
// and readbacks are unavailable.
class WebGPUCore : public Core {
 public:
  explicit WebGPUCore(const Settings &settings);

  static void SetDevice(wgpu::Device device);

  BackendAPI API() const override {
    return BACKEND_API_WEBGPU;
  }

  int CreateBuffer(size_t size, BufferType type, double_ptr<Buffer> pp_buffer) override;

  int CreateImage(int width, int height, ImageFormat format, double_ptr<Image> pp_image) override;

  int CreateSampler(const SamplerInfo &info, double_ptr<Sampler> pp_sampler) override;

  int CreateWindowObject(int width,
                         int height,
                         const std::string &title,
                         bool fullscreen,
                         bool resizable,
                         double_ptr<Window> pp_window) override;

  int CreateShader(const std::string &source_code,
                   const std::string &entry_point,
                   const std::string &target,
                   double_ptr<Shader> pp_shader) override;

  int CreateShader(const VirtualFileSystem &vfs,
                   const std::string &source_file,
                   const std::string &entry_point,
                   const std::string &target,
                   double_ptr<Shader> pp_shader) override;

  int CreateShader(const VirtualFileSystem &vfs,
                   const std::string &source_file,
                   const std::string &entry_point,
                   const std::string &target,
                   const std::vector<std::string> &args,
                   double_ptr<Shader> pp_shader) override;

  int CreateProgram(const std::vector<ImageFormat> &color_formats,
                    ImageFormat depth_format,
                    double_ptr<Program> pp_program) override;

  int CreateComputeProgram(Shader *compute_shader, double_ptr<ComputeProgram> pp_program) override;

  int CreateCommandContext(double_ptr<CommandContext> pp_command_context) override;

  int CreateBottomLevelAccelerationStructure(BufferRange aabb_buffer,
                                             uint32_t stride,
                                             uint32_t num_aabb,
                                             RayTracingGeometryFlag flags,
                                             double_ptr<AccelerationStructure> pp_blas) override;

  int CreateBottomLevelAccelerationStructure(BufferRange vertex_buffer,
                                             BufferRange index_buffer,
                                             uint32_t num_vertex,
                                             uint32_t stride,
                                             uint32_t num_primitive,
                                             RayTracingGeometryFlag flags,
                                             double_ptr<AccelerationStructure> pp_blas) override;

  int CreateBottomLevelAccelerationStructure(Buffer *vertex_buffer,
                                             Buffer *index_buffer,
                                             uint32_t stride,
                                             double_ptr<AccelerationStructure> pp_blas) override;

  int CreateTopLevelAccelerationStructure(const std::vector<RayTracingInstance> &instances,
                                          double_ptr<AccelerationStructure> pp_tlas) override;

  int CreateRayTracingProgram(double_ptr<RayTracingProgram> pp_program) override;

  int SubmitCommandContext(CommandContext *p_command_context) override;

  int GetPhysicalDeviceProperties(PhysicalDeviceProperties *p_physical_device_properties = nullptr) override;

  int InitializeLogicalDevice(int device_index) override;

  void WaitGPU() override {
  }

  uint32_t WaveSize() const override {
    return 32;
  }

  uint32_t CurrentFrame() const override {
    return frame_;
  }

  const wgpu::Device &Device() const {
    return device_;
  }

  const wgpu::Queue &Queue() const {
    return queue_;
  }

 private:
  wgpu::Device device_;
  wgpu::Queue queue_;
  uint32_t frame_ = 0;
};

}  // namespace grassland::graphics::backend
