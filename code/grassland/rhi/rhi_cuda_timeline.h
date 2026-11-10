#pragma once
#include "grassland/rhi/rhi_timeline.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
#include <cuda_runtime_api.h>

namespace grassland::rhi {
// Device-owned, like ITimeline. Finish all CUDA use before destroying the Device.
// The caller orders signals with strictly increasing values, including across streams.
class ICUDATimeline : public ITimeline {
 public:
  // Enqueue a GPU wait without blocking the CPU. stream == nullptr is CUDA's default stream.
  virtual bool Wait(cudaStream_t stream, Event event) = 0;
  // Enqueue a signal and return its graphics-waitable event; an empty event means failure.
  virtual Event Signal(cudaStream_t stream, uint64_t value) = 0;

 protected:
  ~ICUDATimeline() override = default;
};
}  // namespace grassland::rhi
#endif
