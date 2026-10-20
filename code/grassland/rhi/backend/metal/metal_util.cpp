// RHI supplies its own metal-cpp symbols and never links the old graphics runtime.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "grassland/rhi/backend/metal/metal_util.h"

#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

namespace grassland::rhi::backend::metal {
bool SupportsImageStorage(ImageFormat format) {
  // Of the currently supported formats, RGBA8Unorm supports shader writes
  // on all supported Metal GPU families; BGRA8Unorm is read-only.
  return format == IMAGE_FORMAT_RGBA8_UNORM;
}

MTL::PixelFormat NativeImageFormat(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_RGBA8_UNORM:
      return MTL::PixelFormatRGBA8Unorm;
    case IMAGE_FORMAT_BGRA8_UNORM:
      return MTL::PixelFormatBGRA8Unorm;
    default:
      return MTL::PixelFormatInvalid;
  }
}
}  // namespace grassland::rhi::backend::metal
