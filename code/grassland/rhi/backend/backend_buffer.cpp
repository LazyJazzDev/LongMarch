#include "grassland/rhi/backend/backend_buffer.h"

namespace grassland::rhi::backend {
Buffer::Buffer(Device *device, const BufferDesc &desc) : device_(device), desc_(desc) {
}

BufferDesc Buffer::GetDesc() const {
  return desc_;
}

Device *Buffer::Owner() const {
  return device_;
}
}  // namespace grassland::rhi::backend
