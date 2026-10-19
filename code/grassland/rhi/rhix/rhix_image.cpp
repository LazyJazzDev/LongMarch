#include "grassland/rhi/rhix/rhix_image.h"

namespace grassland::rhix {
Image::Image() noexcept = default;

Image::Image(rhi::IImage *object) noexcept : ObjectContainer(object) {
}

rhi::IImage *Image::Get() const noexcept {
  return static_cast<rhi::IImage *>(GetObject());
}

rhi::ImageDesc Image::GetDesc() const {
#if !defined(NDEBUG)
  if (!*this)
    return rhi::ImageDesc{};
#endif
  return Get()->GetDesc();
}

Image::operator bool() const noexcept {
  return ObjectContainer::operator bool() && Get()->IsValid();
}
}  // namespace grassland::rhix
