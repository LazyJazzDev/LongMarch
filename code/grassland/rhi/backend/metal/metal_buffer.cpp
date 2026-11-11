#include "grassland/rhi/backend/metal/metal_buffer.h"

#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {
MetalBuffer::MetalBuffer(MetalDevice *device, const BufferDesc &desc) : Buffer(desc), device_(device) {
}

Device *MetalBuffer::Owner() const {
  return device_;
}

}  // namespace grassland::rhi::backend::metal
