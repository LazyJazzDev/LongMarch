#pragma once
#include "grassland/rhi/rhi_buffer.h"

#if defined(LONGMARCH_CUDA_RUNTIME)
LONGMARCH_RHI_INTERFACE(grassland::rhi::ICUDABuffer, grassland::rhi::IBuffer, "84bb4846-d7f3-478c-b73b-d69d1522ef0f")

namespace grassland::rhi {
// Fixed graphics allocation imported and mapped once into CUDA.
// Use ICUDATimeline and submission Events to order CUDA and graphics access.
// Graphics submissions acquire/release native external-resource ownership automatically.
// Complete all CUDA work before releasing the last reference. The pointer is borrowed
// and stays unchanged for the buffer's lifetime; never cudaFree it in caller code.
class ICUDABuffer : public IBuffer {
 public:
  virtual void *GetDevicePointer() const = 0;

 protected:
  ~ICUDABuffer() override;
};
}  // namespace grassland::rhi
#endif
