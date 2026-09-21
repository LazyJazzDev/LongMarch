#pragma once
#include "sparkium/backend/device.h"

namespace sparkium::backend {
class CpuDevice final : public Device {
 public:
  explicit CpuDevice(const Settings &settings) : Device(settings) {
  }

  int CreateBuffer(size_t, BufferType, double_ptr<Buffer>) override;
  int CreateImage(int, int, ImageFormat, double_ptr<Image>) override;
  int CreateSampler(const SamplerInfo &, double_ptr<Sampler>) override;
  int CreateShader(const std::string &, const std::string &, const std::string &, double_ptr<Shader>) override;
  int CreateShader(const VirtualFileSystem &,
                   const std::string &,
                   const std::string &,
                   const std::string &,
                   double_ptr<Shader>) override;
  int CreateShader(const VirtualFileSystem &,
                   const std::string &,
                   const std::string &,
                   const std::string &,
                   const std::vector<std::string> &,
                   double_ptr<Shader>) override;
  int CreateComputeProgram(Shader *, double_ptr<ComputeProgram>) override;
  int CreateCommandContext(double_ptr<CommandContext>) override;
  int SubmitCommandContext(CommandContext *) override;

  uint32_t WaveSize() const override {
    return 1;
  }

  uint32_t CurrentFrame() const override {
    return 0;
  }

  int CreateProgram(const std::vector<ImageFormat> &, ImageFormat, double_ptr<Program>) override;
  int CreateBottomLevelAccelerationStructure(BufferRange,
                                             uint32_t,
                                             uint32_t,
                                             RayTracingGeometryFlag,
                                             double_ptr<AccelerationStructure>) override;
  int CreateBottomLevelAccelerationStructure(BufferRange,
                                             BufferRange,
                                             uint32_t,
                                             uint32_t,
                                             uint32_t,
                                             RayTracingGeometryFlag,
                                             double_ptr<AccelerationStructure>) override;
  int CreateBottomLevelAccelerationStructure(Buffer *, Buffer *, uint32_t, double_ptr<AccelerationStructure>) override;
  int CreateTopLevelAccelerationStructure(const std::vector<RayTracingInstance> &,
                                          double_ptr<AccelerationStructure>) override;
  int CreateRayTracingProgram(double_ptr<RayTracingProgram>) override;

  RenderBackend API() const override {
    return RenderBackend::CPU;
  }

  int GetPhysicalDeviceProperties(PhysicalDeviceProperties *properties = nullptr) override;
  int InitializeLogicalDevice(int index) override;

  void WaitGPU() override {
  }
};
}  // namespace sparkium::backend
