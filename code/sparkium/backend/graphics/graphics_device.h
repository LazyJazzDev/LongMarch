#pragma once
#include "sparkium/backend/device.h"

namespace sparkium::backend {
class GraphicsDevice final : public Device {
 public:
  explicit GraphicsDevice(graphics::Core *core) : Device({core->FramesInFlight(), core->DebugEnabled()}), core_(core) {
  }

  explicit GraphicsDevice(std::unique_ptr<graphics::Core> core) : GraphicsDevice(core.get()) {
    owned_ = std::move(core);
  }

  RenderBackend API() const override {
    return RenderBackend::Graphics;
  }

  graphics::Core *GraphicsCore() const override {
    return core_;
  }

  uint32_t CurrentFrame() const override {
    return core_->CurrentFrame();
  }

  bool DeviceRayTracingSupport() const override {
    return core_->DeviceRayTracingSupport();
  }

  bool DeviceRayQuerySupport() const override {
    return core_->DeviceRayQuerySupport();
  }

  std::string DeviceName() const override {
    return core_->DeviceName();
  }

  int CreateBuffer(size_t size, BufferType type, double_ptr<Buffer> pp_buffer) override {
    return core_->CreateBuffer(size, type, pp_buffer);
  }

  int CreateImage(int width, int height, ImageFormat format, double_ptr<Image> pp_image) override {
    return core_->CreateImage(width, height, format, pp_image);
  }

  int CreateSampler(const SamplerInfo &info, double_ptr<Sampler> pp_sampler) override {
    return core_->CreateSampler(info, pp_sampler);
  }

  int CreateShader(const std::string &source_code,
                   const std::string &entry_point,
                   const std::string &target,
                   double_ptr<Shader> pp_shader) override {
    return core_->CreateShader(source_code, entry_point, target, pp_shader);
  }

  int CreateShader(const VirtualFileSystem &vfs,
                   const std::string &source_file,
                   const std::string &entry_point,
                   const std::string &target,
                   double_ptr<Shader> pp_shader) override {
    return core_->CreateShader(vfs, source_file, entry_point, target, pp_shader);
  }

  int CreateShader(const VirtualFileSystem &vfs,
                   const std::string &source_file,
                   const std::string &entry_point,
                   const std::string &target,
                   const std::vector<std::string> &args,
                   double_ptr<Shader> pp_shader) override {
    return core_->CreateShader(vfs, source_file, entry_point, target, args, pp_shader);
  }

  int CreateProgram(const std::vector<ImageFormat> &color_formats,
                    ImageFormat depth_format,
                    double_ptr<Program> pp_program) override {
    return core_->CreateProgram(color_formats, depth_format, pp_program);
  }

  int CreateComputeProgram(Shader *compute_shader, double_ptr<ComputeProgram> pp_program) override {
    return core_->CreateComputeProgram(compute_shader, pp_program);
  }

  int CreateCommandContext(double_ptr<CommandContext> pp_command_context) override {
    return core_->CreateCommandContext(pp_command_context);
  }

  int CreateBottomLevelAccelerationStructure(BufferRange aabb_buffer,
                                             uint32_t stride,
                                             uint32_t num_aabb,
                                             RayTracingGeometryFlag flags,
                                             double_ptr<AccelerationStructure> pp_blas) override {
    return core_->CreateBottomLevelAccelerationStructure(aabb_buffer, stride, num_aabb, flags, pp_blas);
  }

  int CreateBottomLevelAccelerationStructure(BufferRange vertex_buffer,
                                             BufferRange index_buffer,
                                             uint32_t num_vertex,
                                             uint32_t stride,
                                             uint32_t num_primitive,
                                             RayTracingGeometryFlag flags,
                                             double_ptr<AccelerationStructure> pp_blas) override {
    return core_->CreateBottomLevelAccelerationStructure(vertex_buffer, index_buffer, num_vertex, stride, num_primitive,
                                                         flags, pp_blas);
  }

  int CreateBottomLevelAccelerationStructure(Buffer *vertex_buffer,
                                             Buffer *index_buffer,
                                             uint32_t stride,
                                             double_ptr<AccelerationStructure> pp_blas) override {
    return core_->CreateBottomLevelAccelerationStructure(vertex_buffer, index_buffer, stride, pp_blas);
  }

  int CreateTopLevelAccelerationStructure(const std::vector<RayTracingInstance> &instances,
                                          double_ptr<AccelerationStructure> pp_tlas) override {
    return core_->CreateTopLevelAccelerationStructure(instances, pp_tlas);
  }

  int CreateRayTracingProgram(double_ptr<RayTracingProgram> pp_program) override {
    return core_->CreateRayTracingProgram(pp_program);
  }

  int SubmitCommandContext(CommandContext *p_command_context) override {
    return core_->SubmitCommandContext(p_command_context);
  }

  int GetPhysicalDeviceProperties(PhysicalDeviceProperties *p_physical_device_properties = nullptr) override {
    return core_->GetPhysicalDeviceProperties(p_physical_device_properties);
  }

  int InitializeLogicalDevice(int device_index) override {
    return core_->InitializeLogicalDevice(device_index);
  }

  void WaitGPU() override {
    return core_->WaitGPU();
  }

  uint32_t WaveSize() const override {
    return core_->WaveSize();
  }

 private:
  std::unique_ptr<graphics::Core> owned_;
  graphics::Core *core_;
};
}  // namespace sparkium::backend
