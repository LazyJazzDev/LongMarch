#include "grassland/rhi/backend/backend_buffer.h"

namespace grassland::rhi::backend {
Buffer::Buffer(const BufferDesc &desc) : desc_(desc) {
}

BufferDesc Buffer::GetDesc() const {
  return desc_;
}

}  // namespace grassland::rhi::backend
