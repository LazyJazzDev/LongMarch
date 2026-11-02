#pragma once
#include "grassland/rhi/rhi_cuda_interop.h"
#include "grassland/rhi/rhix/rhix_buffer.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
namespace grassland::rhix {
class CUDABuffer : public Buffer {
 public:
  CUDABuffer() noexcept = default;
  CUDABuffer(rhi::ICUDABuffer *object) noexcept;
  rhi::ICUDABuffer *Get() const noexcept;
  void *GetDevicePointer() const;
};

class CUDAInteropTimeline : public ObjectContainer {
 public:
  CUDAInteropTimeline() noexcept = default;
  CUDAInteropTimeline(rhi::ICUDAInteropTimeline *object) noexcept;
  rhi::ICUDAInteropTimeline *Get() const noexcept;
  bool SignalCUDA(cudaStream_t stream, uint64_t value) const;
  bool WaitCUDA(cudaStream_t stream, uint64_t value) const;
};

struct CUDAInteropPoint {
  CUDAInteropTimeline timeline;
  uint64_t value{};
};
}  // namespace grassland::rhix
#endif
