#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_buffer.h"

namespace grassland::rhi::backend::metal {
class MetalDevice;

class MetalBuffer : public Buffer {
 public:
  BufferDesc GetDesc() const override {
    return Description();
  }

  virtual MTL::Buffer *NativeBuffer() const = 0;

 protected:
  MetalBuffer(MetalDevice *device, const BufferDesc &desc);
};
}  // namespace grassland::rhi::backend::metal
