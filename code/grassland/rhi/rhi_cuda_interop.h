#pragma once
#include "grassland/rhi/rhi_buffer.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
#include <cuda_runtime_api.h>

namespace grassland::rhi {
// Fixed allocation, mapped once. Synchronize both APIs before releasing it.
class ICUDABuffer : public virtual IBuffer {
 public:
  virtual void *GetDevicePointer() const = 0;

 protected:
  ~ICUDABuffer() override = default;
};

// Values must increase in GPU execution order. Keep alive until all CUDA use completes.
class ICUDAInteropTimeline : public IObject {
 public:
  virtual bool SignalCUDA(cudaStream_t stream, uint64_t value) = 0;
  virtual bool WaitCUDA(cudaStream_t stream, uint64_t value) = 0;

 protected:
  ~ICUDAInteropTimeline() override = default;
};

struct CUDAInteropPoint {
  ICUDAInteropTimeline *timeline{};
  uint64_t value{};
};
}  // namespace grassland::rhi
#endif
