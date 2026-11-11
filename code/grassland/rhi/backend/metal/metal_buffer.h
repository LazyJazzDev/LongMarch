#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_buffer.h"

namespace grassland::rhi::backend::metal {
class MetalDevice;

class MetalBuffer : public Buffer {
 public:
  LONGMARCH_RHI_QUERY(MetalBuffer, Buffer)

  virtual MTL::Buffer *NativeBuffer() const = 0;

  Device *Owner() const override;

 protected:
  MetalDevice *device_;
  MetalBuffer(MetalDevice *device, const BufferDesc &desc);
};
}  // namespace grassland::rhi::backend::metal
