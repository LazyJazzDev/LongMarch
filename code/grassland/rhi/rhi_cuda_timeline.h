#pragma once
#include "grassland/rhi/rhi_timeline.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
#include <cuda_runtime_api.h>

#include <vector>

LONGMARCH_RHI_INTERFACE(grassland::rhi::ICUDATimeline,
                        grassland::rhi::ITimeline,
                        "6a47b290-e78f-4356-8cce-c4e5c626feba")

namespace grassland::rhi {
// Owned reference; finish pending GPU use before releasing it, and release it before the Device.
// Events do not retain their timelines; keep every source alive until CUDA finishes waiting.
// Queue timeline values belong to the queue: never signal a queue timeline from CUDA.
// The caller orders signals with strictly increasing values, including across streams.
class ICUDATimeline : public ITimeline {
 public:
  // Enqueue waits for all events from this device without blocking the CPU.
  // Empty lists succeed; stream == nullptr is CUDA's default stream.
  virtual bool Wait(cudaStream_t stream, const std::vector<Event> &events) = 0;
  // Enqueue a signal and return its graphics-waitable event; an empty event means failure.
  virtual Event Signal(cudaStream_t stream, uint64_t value) = 0;

 protected:
  ~ICUDATimeline() override = default;
};
}  // namespace grassland::rhi
#endif
