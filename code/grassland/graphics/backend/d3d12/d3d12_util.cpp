#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {
DXGI_FORMAT ImageFormatToDXGIFormat(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_B8G8R8A8_UNORM:
      return DXGI_FORMAT_B8G8R8A8_UNORM;
    case IMAGE_FORMAT_R32G32B32A32_SFLOAT:
      return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case IMAGE_FORMAT_R32G32B32_SFLOAT:
      return DXGI_FORMAT_R32G32B32_FLOAT;
    case IMAGE_FORMAT_R32G32_SFLOAT:
      return DXGI_FORMAT_R32G32_FLOAT;
    case IMAGE_FORMAT_R32_SFLOAT:
      return DXGI_FORMAT_R32_FLOAT;
    case IMAGE_FORMAT_D32_SFLOAT:
      return DXGI_FORMAT_D32_FLOAT;
    default:
      return DXGI_FORMAT_UNKNOWN;
  }
}

DXGI_FORMAT InputTypeToDXGIFormat(InputType type) {
  switch (type) {
    case INPUT_TYPE_INT:
      return DXGI_FORMAT_R32_SINT;
    case INPUT_TYPE_UINT:
      return DXGI_FORMAT_R32_UINT;
    case INPUT_TYPE_FLOAT:
      return DXGI_FORMAT_R32_FLOAT;
    case INPUT_TYPE_INT2:
      return DXGI_FORMAT_R32G32_SINT;
    case INPUT_TYPE_UINT2:
      return DXGI_FORMAT_R32G32_UINT;
    case INPUT_TYPE_FLOAT2:
      return DXGI_FORMAT_R32G32_FLOAT;
    case INPUT_TYPE_INT3:
      return DXGI_FORMAT_R32G32B32_SINT;
    case INPUT_TYPE_UINT3:
      return DXGI_FORMAT_R32G32B32_UINT;
    case INPUT_TYPE_FLOAT3:
      return DXGI_FORMAT_R32G32B32_FLOAT;
    case INPUT_TYPE_INT4:
      return DXGI_FORMAT_R32G32B32A32_SINT;
    case INPUT_TYPE_UINT4:
      return DXGI_FORMAT_R32G32B32A32_UINT;
    case INPUT_TYPE_FLOAT4:
      return DXGI_FORMAT_R32G32B32A32_FLOAT;
    default:
      return DXGI_FORMAT_UNKNOWN;
  }
}
}  // namespace grassland::graphics::backend
