#pragma once
#include "sparkium/backend/common/compute_device.h"
#ifdef SPARKIUM_OPTIX_ENABLED
#include "sparkium/backend/cuda/optix_device.h"
#endif
namespace sparkium::backend {
class CudaDevice final : public ComputeDevice {
 public:
  explicit CudaDevice(const Settings &settings) : ComputeDevice(settings) {
  }

  ~CudaDevice() override;

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
  OptixDevice *Optix() const override {
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
