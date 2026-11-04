#pragma once
#include "grassland/rhi/rhi_buffer.h"

#if defined(LONGMARCH_CUDA_RUNTIME)
namespace grassland::rhi {
// Fixed graphics allocation imported and mapped once into CUDA.
// This initial API supports CUDA access only; graphics commands/handoffs are deferred.
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
