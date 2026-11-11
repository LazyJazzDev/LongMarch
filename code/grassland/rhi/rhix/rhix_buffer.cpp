#include "grassland/rhi/rhix/rhix_buffer.h"

namespace grassland::rhix {
Buffer::Buffer() noexcept = default;

Buffer::Buffer(rhi::IBuffer *object) noexcept : ObjectContainer(object) {
}

rhi::IBuffer *Buffer::Get() const noexcept {
  return static_cast<rhi::IBuffer *>(GetObject());
}

rhi::BufferDesc Buffer::GetDesc() const {
  return Get()->GetDesc();
}
}  // namespace grassland::rhix
