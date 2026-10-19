#pragma once
#include "grassland/rhi/rhi.h"
#include "grassland/rhi/rhi_util.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
class Image : public ObjectContainer {
 public:
  Image(rhi::IImage *object) noexcept;
  rhi::IImage *Get() const noexcept;
  Image() noexcept;
  rhi::ImageDesc GetDesc() const;
  // Acquired swapchain images are single-use. Resize or mode changes invalidate
  // old frames; callers must release them before requesting another frame.
  explicit operator bool() const noexcept;
};

}  // namespace grassland::rhix
