#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::metal {
class MetalBuffer;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::MetalBuffer)
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
