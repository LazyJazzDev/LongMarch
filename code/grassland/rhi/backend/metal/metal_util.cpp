// RHI supplies its own metal-cpp symbols and never links the old graphics runtime.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "grassland/rhi/backend/metal/metal_util.h"

#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

namespace grassland::rhi::backend::metal {
MTL::BlendFactor NativeBlendFactor(BlendFactor factor, bool alpha) {
  // Alpha equations use the alpha component of color factors as well.
  if (alpha && factor >= BLEND_FACTOR_SRC_COLOR && factor <= BLEND_FACTOR_ONE_MINUS_DST_COLOR)
    factor = static_cast<BlendFactor>(factor + BLEND_FACTOR_SRC_ALPHA - BLEND_FACTOR_SRC_COLOR);
  const MTL::BlendFactor factors[] = {MTL::BlendFactorZero,
                                      MTL::BlendFactorOne,
                                      MTL::BlendFactorSourceColor,
                                      MTL::BlendFactorOneMinusSourceColor,
                                      MTL::BlendFactorDestinationColor,
                                      MTL::BlendFactorOneMinusDestinationColor,
                                      MTL::BlendFactorSourceAlpha,
                                      MTL::BlendFactorOneMinusSourceAlpha,
                                      MTL::BlendFactorDestinationAlpha,
                                      MTL::BlendFactorOneMinusDestinationAlpha};
  return factors[factor];
}

MTL::BlendOperation NativeBlendOp(BlendOp op) {
  const MTL::BlendOperation operations[] = {MTL::BlendOperationAdd, MTL::BlendOperationSubtract,
                                            MTL::BlendOperationReverseSubtract, MTL::BlendOperationMin,
                                            MTL::BlendOperationMax};
  return operations[op];
}

bool SupportsImageStorage(ImageFormat format) {
  // These formats support shader writes on all supported Metal GPU families.
  return format == IMAGE_FORMAT_RGBA8_UNORM || format == IMAGE_FORMAT_RGBA16_FLOAT;
}

MTL::PixelFormat NativeImageFormat(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_RGBA8_UNORM:
      return MTL::PixelFormatRGBA8Unorm;
    case IMAGE_FORMAT_BGRA8_UNORM:
      return MTL::PixelFormatBGRA8Unorm;
    case IMAGE_FORMAT_RGBA16_FLOAT:
      return MTL::PixelFormatRGBA16Float;
    case IMAGE_FORMAT_RGB10A2_UNORM:
      return MTL::PixelFormatRGB10A2Unorm;
    case IMAGE_FORMAT_BGR10A2_UNORM:
      return MTL::PixelFormatBGR10A2Unorm;
    case IMAGE_FORMAT_D32_FLOAT:
      return MTL::PixelFormatDepth32Float;
    default:
      return MTL::PixelFormatInvalid;
  }
}
}  // namespace grassland::rhi::backend::metal
