#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_cuda_timeline.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::IBackendCUDATimeline,
                        grassland::rhi::IObject,
                        "e08fb6dd-cb83-4b2b-b745-a4d58868481e")

namespace grassland::rhi::backend {
class IBackendCUDATimeline : public IObject {
 public:
  virtual const void *CUDAOwner() const = 0;
  virtual cudaExternalSemaphore_t CUDASemaphore() const = 0;
};

class CUDAExternalSemaphore final {
 public:
  explicit CUDAExternalSemaphore(int device) : device_(device) {
  }

  CUDAExternalSemaphore(const CUDAExternalSemaphore &) = delete;
  CUDAExternalSemaphore &operator=(const CUDAExternalSemaphore &) = delete;
  ~CUDAExternalSemaphore();
  bool Import(const cudaExternalSemaphoreHandleDesc &desc);
  void Reset();

  cudaExternalSemaphore_t Get() const {
    return semaphore_;
  }

  bool Wait(const void *owner, cudaStream_t stream, const std::vector<Event> &events) const;
  bool Signal(cudaStream_t stream, uint64_t value) const;

 private:
  int device_;
  cudaExternalSemaphore_t semaphore_{};
};
}  // namespace grassland::rhi::backend
#endif
