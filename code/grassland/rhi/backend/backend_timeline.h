#pragma once
#include "grassland/rhi/rhi_cuda_timeline.h"
#include "grassland/rhi/rhi_timeline.h"

namespace grassland::rhi::backend {
class Device;

class Timeline : public
#if defined(LONGMARCH_CUDA_RUNTIME)
                 ICUDATimeline
#else
                 ITimeline
#endif
{
 public:
  explicit Timeline(Device *device) : device_(device) {
  }

  ~Timeline() override = default;
#if defined(LONGMARCH_CUDA_RUNTIME)
  bool Wait(cudaStream_t stream, Event event) override;
  Event Signal(cudaStream_t stream, uint64_t value) override;
#endif

  Device *Owner() const {
    return device_;
  }

 protected:
#if defined(LONGMARCH_CUDA_RUNTIME)
  bool ImportCUDA(const cudaExternalSemaphoreHandleDesc &desc);
  void ReleaseCUDA();
#endif

 private:
  Device *device_;
#if defined(LONGMARCH_CUDA_RUNTIME)
  cudaExternalSemaphore_t cuda_semaphore_{};
#endif
};

}  // namespace grassland::rhi::backend
