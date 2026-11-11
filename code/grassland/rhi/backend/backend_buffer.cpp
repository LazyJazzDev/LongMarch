#include "grassland/rhi/backend/backend_buffer.h"

namespace grassland::rhi::backend {
Buffer::Buffer(const BufferDesc &desc) : desc_(desc) {
}

BufferDesc Buffer::GetDesc() const {
  return desc_;
}

}  // namespace grassland::rhi::backend

LONGMARCH_RHI_UUID(grassland::rhi::backend::Buffer, "a86fae4e-f66b-43e7-89b4-3d5c431bf0d3")
