#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_buffer.h"

namespace grassland::rhi::backend {
class Device;

class Buffer : public Object<IBuffer> {
 public:
  BufferDesc Description() const;
  Device *Owner() const;

 protected:
  Buffer(Device *device, const BufferDesc &desc);
  Device *device_;
  BufferDesc desc_;
};
}  // namespace grassland::rhi::backend
