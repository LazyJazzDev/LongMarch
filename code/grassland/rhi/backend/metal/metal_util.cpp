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
  return IsColorFormat(format) && NativeImageFormat(format) != MTL::PixelFormatInvalid &&
         format != IMAGE_FORMAT_B8G8R8A8_UNORM && format != IMAGE_FORMAT_R10G10B10A2_UNORM &&
         format != IMAGE_FORMAT_B10G10R10A2_UNORM;
}

MTL::PixelFormat NativeImageFormat(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_R8_UINT:
      return MTL::PixelFormatR8Uint;
    case IMAGE_FORMAT_R8_INT:
      return MTL::PixelFormatR8Sint;
    case IMAGE_FORMAT_R8_UNORM:
      return MTL::PixelFormatR8Unorm;
    case IMAGE_FORMAT_R8G8_UINT:
      return MTL::PixelFormatRG8Uint;
    case IMAGE_FORMAT_R8G8_INT:
      return MTL::PixelFormatRG8Sint;
    case IMAGE_FORMAT_R8G8_UNORM:
      return MTL::PixelFormatRG8Unorm;
    case IMAGE_FORMAT_R8G8B8A8_UINT:
      return MTL::PixelFormatRGBA8Uint;
    case IMAGE_FORMAT_R8G8B8A8_INT:
      return MTL::PixelFormatRGBA8Sint;
    case IMAGE_FORMAT_R8G8B8A8_UNORM:
      return MTL::PixelFormatRGBA8Unorm;
    case IMAGE_FORMAT_R32_UINT:
      return MTL::PixelFormatR32Uint;
    case IMAGE_FORMAT_R32_INT:
      return MTL::PixelFormatR32Sint;
    case IMAGE_FORMAT_R32_SFLOAT:
      return MTL::PixelFormatR32Float;
    case IMAGE_FORMAT_R32G32_UINT:
      return MTL::PixelFormatRG32Uint;
    case IMAGE_FORMAT_R32G32_INT:
      return MTL::PixelFormatRG32Sint;
    case IMAGE_FORMAT_R32G32_SFLOAT:
      return MTL::PixelFormatRG32Float;
    case IMAGE_FORMAT_R32G32B32A32_UINT:
      return MTL::PixelFormatRGBA32Uint;
    case IMAGE_FORMAT_R32G32B32A32_INT:
      return MTL::PixelFormatRGBA32Sint;
    case IMAGE_FORMAT_R32G32B32A32_SFLOAT:
      return MTL::PixelFormatRGBA32Float;
    case IMAGE_FORMAT_B8G8R8A8_UNORM:
      return MTL::PixelFormatBGRA8Unorm;
    case IMAGE_FORMAT_R16G16B16A16_SFLOAT:
      return MTL::PixelFormatRGBA16Float;
    case IMAGE_FORMAT_R10G10B10A2_UNORM:
      return MTL::PixelFormatRGB10A2Unorm;
    case IMAGE_FORMAT_B10G10R10A2_UNORM:
      return MTL::PixelFormatBGR10A2Unorm;
    case IMAGE_FORMAT_D16_UNORM:
      return MTL::PixelFormatDepth16Unorm;
    case IMAGE_FORMAT_D32_SFLOAT:
      return MTL::PixelFormatDepth32Float;
    case IMAGE_FORMAT_D24_UNORM_S8_UINT:
      return MTL::PixelFormatDepth32Float_Stencil8;
    default:
      return MTL::PixelFormatInvalid;
  }
}
}  // namespace grassland::rhi::backend::metal
