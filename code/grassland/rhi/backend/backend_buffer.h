#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_buffer.h"

namespace grassland::rhi::backend {
class Device;

// Share one IBuffer interface with the optional CUDA specialization.
#if defined(LONGMARCH_CUDA_RUNTIME)
class BufferInterface : public virtual IBuffer {};
#else
using BufferInterface = IBuffer;
#endif

class Buffer : public Object<BufferInterface> {
 public:
  BufferDesc GetDesc() const override;
  Device *Owner() const;

 protected:
  Buffer(Device *device, const BufferDesc &desc);
  Device *device_;
  BufferDesc desc_;
};
}  // namespace grassland::rhi::backend
