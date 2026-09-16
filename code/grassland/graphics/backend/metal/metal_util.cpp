#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "grassland/graphics/backend/metal/metal_util.h"

#include <QuartzCore/QuartzCore.hpp>
#include <stdexcept>

namespace grassland::graphics::backend {

void MetalCheck(const void *object, NS::Error *error, const char *operation) {
  if (!object)
    throw std::runtime_error(std::string(operation) + ": " +
                             (error ? error->localizedDescription()->utf8String() : "Metal returned null"));
}

MTL::PixelFormat MetalFormat(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_UNDEFINED:
      return MTL::PixelFormatInvalid;
    case IMAGE_FORMAT_B8G8R8A8_UNORM:
      return MTL::PixelFormatBGRA8Unorm;
    case IMAGE_FORMAT_R8G8B8A8_UNORM:
      return MTL::PixelFormatRGBA8Unorm;
    case IMAGE_FORMAT_R32G32B32A32_SFLOAT:
    case IMAGE_FORMAT_R32G32B32_SFLOAT:
      return MTL::PixelFormatRGBA32Float;
    case IMAGE_FORMAT_R32G32_SFLOAT:
      return MTL::PixelFormatRG32Float;
    case IMAGE_FORMAT_R32_SFLOAT:
      return MTL::PixelFormatR32Float;
    case IMAGE_FORMAT_D32_SFLOAT:
      return MTL::PixelFormatDepth32Float;
    case IMAGE_FORMAT_R16G16B16A16_SFLOAT:
      return MTL::PixelFormatRGBA16Float;
    case IMAGE_FORMAT_R32_UINT:
      return MTL::PixelFormatR32Uint;
    case IMAGE_FORMAT_R32_SINT:
      return MTL::PixelFormatR32Sint;
  }
  throw std::runtime_error("unsupported Metal image format");
}

}  // namespace grassland::graphics::backend
