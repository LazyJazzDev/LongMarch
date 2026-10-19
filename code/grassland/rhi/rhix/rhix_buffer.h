#pragma once
#include "grassland/rhi/rhi_buffer.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
class Buffer : public ObjectContainer {
 public:
  Buffer() noexcept;
  Buffer(rhi::IBuffer *object) noexcept;
  rhi::IBuffer *Get() const noexcept;
  rhi::BufferDesc GetDesc() const;
};
}  // namespace grassland::rhix
