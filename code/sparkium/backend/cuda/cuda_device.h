#pragma once
#include "sparkium/backend/device.h"
#ifdef SPARKIUM_OPTIX_ENABLED
#include "sparkium/backend/cuda/optix_device.h"
#endif
namespace sparkium::backend {
class CudaDevice final : public Device {
 public:
  explicit CudaDevice(const Settings &settings) : Device(settings) {
  }

  ~CudaDevice() override;

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
    return 32;
  }

  uint32_t CurrentFrame() const override {
    return 0;
  }

  int CreateProgram(const std::vector<ImageFormat> &, ImageFormat, double_ptr<Program>) override;
  int CreateRayTracingProgram(double_ptr<RayTracingProgram>) override;

  RenderBackend API() const override {
    return RenderBackend::CUDA;
  }

  int GetPhysicalDeviceProperties(PhysicalDeviceProperties *properties = nullptr) override;
  int InitializeLogicalDevice(int index) override;
  void WaitGPU() override;
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

 protected:
  OptixDevice *Optix() const {
#ifdef SPARKIUM_OPTIX_ENABLED
    return optix_.get();
#else
    return nullptr;
#endif
  }

 private:
  void *cuda_context_{};
  int device_index_{-1};
#ifdef SPARKIUM_OPTIX_ENABLED
  std::unique_ptr<OptixDevice> optix_;
#endif
};
}  // namespace sparkium::backend
