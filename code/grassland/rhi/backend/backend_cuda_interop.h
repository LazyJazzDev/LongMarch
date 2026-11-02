#pragma once
#include "grassland/rhi/rhi_cuda_interop.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
#include "grassland/rhi/backend/backend_device.h"

namespace grassland::rhi::backend {
bool CheckCUDA(cudaError_t error, const char *operation);

class CUDADeviceScope {
 public:
  explicit CUDADeviceScope(int device);
  CUDADeviceScope(const CUDADeviceScope &) = delete;
  CUDADeviceScope &operator=(const CUDADeviceScope &) = delete;
  ~CUDADeviceScope();
  explicit operator bool() const;

 private:
  int previous_{-1};
  bool valid_{};
  bool changed_{};
};

class CUDAMapping {
 public:
  explicit CUDAMapping(int device);
  CUDAMapping(const CUDAMapping &) = delete;
  CUDAMapping &operator=(const CUDAMapping &) = delete;
  ~CUDAMapping();
  bool Import(const cudaExternalMemoryHandleDesc &desc, uint64_t size);
  void *Pointer() const;
  bool HasMemory() const;
  void Reset();

 private:
  int device_;
  cudaExternalMemory_t memory_{};
  void *pointer_{};
};

class CUDAInteropTimeline : public Object<ICUDAInteropTimeline> {
 public:
  bool SignalCUDA(cudaStream_t stream, uint64_t value) override;
  bool WaitCUDA(cudaStream_t stream, uint64_t value) override;
  Device *Owner() const;

 protected:
  explicit CUDAInteropTimeline(Device *device);
  ~CUDAInteropTimeline() override;
  bool Import(const cudaExternalSemaphoreHandleDesc &desc);
  void Reset();

 private:
  Device *device_;
  cudaExternalSemaphore_t semaphore_{};
};
}  // namespace grassland::rhi::backend
#endif
