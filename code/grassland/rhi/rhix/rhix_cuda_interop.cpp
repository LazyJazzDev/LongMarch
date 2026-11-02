#include "grassland/rhi/rhix/rhix_cuda_interop.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
namespace grassland::rhix {
CUDABuffer::CUDABuffer(rhi::ICUDABuffer *object) noexcept : Buffer(object) {
}

rhi::ICUDABuffer *CUDABuffer::Get() const noexcept {
  return dynamic_cast<rhi::ICUDABuffer *>(Buffer::Get());
}

void *CUDABuffer::GetDevicePointer() const {
  return Get() ? Get()->GetDevicePointer() : nullptr;
}

CUDAInteropTimeline::CUDAInteropTimeline(rhi::ICUDAInteropTimeline *object) noexcept : ObjectContainer(object) {
}

rhi::ICUDAInteropTimeline *CUDAInteropTimeline::Get() const noexcept {
  return static_cast<rhi::ICUDAInteropTimeline *>(GetObject());
}

bool CUDAInteropTimeline::SignalCUDA(cudaStream_t stream, uint64_t value) const {
  return Get() && Get()->SignalCUDA(stream, value);
}

bool CUDAInteropTimeline::WaitCUDA(cudaStream_t stream, uint64_t value) const {
  return Get() && Get()->WaitCUDA(stream, value);
}
}  // namespace grassland::rhix
#endif
