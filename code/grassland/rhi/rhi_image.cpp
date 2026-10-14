#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/rhi_device.h"

namespace grassland::rhi {
using namespace backend;

ImageDesc Image::GetDesc() const {
  return *this ? Access::As<IImage>(*this).GetDesc() : ImageDesc{};
}

Image::operator bool() const noexcept {
  return object_ && Access::As<IImage>(*this).IsValid();
}

}  // namespace grassland::rhi
