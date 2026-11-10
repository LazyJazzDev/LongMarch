#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_buffer.h"

namespace grassland::rhi::backend {
class Device;

// Backend resource access and allocation metadata; public interfaces own their reference count.
class Buffer {
 public:
  virtual ~Buffer() = default;
  virtual BufferDesc GetDesc() const;
  Device *Owner() const;

 protected:
  Buffer(Device *device, const BufferDesc &desc);
  Device *device_;
  BufferDesc desc_;
};
}  // namespace grassland::rhi::backend
