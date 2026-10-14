#pragma once
#include "grassland/rhi/rhi_util.h"

namespace grassland::rhi {
class Image {
 public:
  ImageDesc GetDesc() const;
  // Acquired swapchain images are single-use. Resize or mode changes invalidate
  // old frames; callers must release them before requesting another frame.
  explicit operator bool() const noexcept;

 private:
  using Interface = backend::IImage;
  ObjectRef object_;
  friend struct backend::Access;
};

}  // namespace grassland::rhi
