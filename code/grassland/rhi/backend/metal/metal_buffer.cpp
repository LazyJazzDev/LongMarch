#include "grassland/rhi/backend/metal/metal_buffer.h"

#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {
MetalBuffer::MetalBuffer(MetalDevice *device, const BufferDesc &desc) : Buffer(desc), device_(device) {
}

Device *MetalBuffer::Owner() const {
  return device_;
}

}  // namespace grassland::rhi::backend::metal

LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::MetalBuffer, "203e6dfb-bb02-4155-900a-b13f43af47bc")
