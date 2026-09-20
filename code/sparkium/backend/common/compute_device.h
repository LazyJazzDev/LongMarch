#pragma once
#include "sparkium/backend/device.h"

namespace sparkium::backend {
using namespace grassland;
using namespace grassland::graphics;
class OptixDevice;

// Headless CPU/CUDA compute, with optional OptiX hardware traversal.
class ComputeDevice : public Device {
 public:
  explicit ComputeDevice(const Settings &settings) : Device(settings) {
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
    return API() == RenderBackend::CPU ? 1 : 32;
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

 protected:
  bool UsesCUDA() const {
    return API() == RenderBackend::CUDA;
  }

  virtual OptixDevice *Optix() const {
    return nullptr;
  }
};
}  // namespace sparkium::backend
