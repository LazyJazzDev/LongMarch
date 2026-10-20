#include "grassland/rhi/backend/metal/metal_buffer.h"

#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {
MetalBuffer::MetalBuffer(MetalDevice *device, const BufferDesc &desc) : Buffer(device, desc) {
}
}  // namespace grassland::rhi::backend::metal
