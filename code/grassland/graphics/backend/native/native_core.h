#pragma once
#include "grassland/graphics/core.h"

namespace grassland::graphics::backend {
// Headless compute devices. No graphics API, swapchain, or hardware RT delegation.
class NativeCore final : public Core {
 public:
  NativeCore(BackendAPI api, const Settings &settings);
  ~NativeCore() override;
  BackendAPI API() const override {
    return api_;
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
  int GetPhysicalDeviceProperties(PhysicalDeviceProperties * = nullptr) override;
  int InitializeLogicalDevice(int) override;
  void WaitGPU() override;
  uint32_t WaveSize() const override {
    return api_ == BACKEND_API_CPU ? 1 : 32;
  }
  uint32_t CurrentFrame() const override {
    return 0;
  }

  int CreateWindowObject(int, int, const std::string &, bool, bool, double_ptr<Window>) override;
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
#if defined(LONGMARCH_CUDA_RUNTIME)
  int CreateCUDABuffer(size_t, double_ptr<CUDABuffer>) override;
  void CUDABeginExecutionBarrier(cudaStream_t = 0) override;
  void CUDAEndExecutionBarrier(cudaStream_t = 0) override;
#endif
 private:
  BackendAPI api_;
  void *cuda_context_{};
  int device_index_{-1};
};
}  // namespace grassland::graphics::backend
