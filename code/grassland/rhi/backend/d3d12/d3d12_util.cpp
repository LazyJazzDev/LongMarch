#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
DXGI_FORMAT NativeImageFormat(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_RGBA8_UNORM:
      return DXGI_FORMAT_R8G8B8A8_UNORM;
    case IMAGE_FORMAT_BGRA8_UNORM:
      return DXGI_FORMAT_B8G8R8A8_UNORM;
    case IMAGE_FORMAT_D32_FLOAT:
      return DXGI_FORMAT_D32_FLOAT;
    case IMAGE_FORMAT_D24_UNORM_S8_UINT:
      return DXGI_FORMAT_D24_UNORM_S8_UINT;
    default:
      return DXGI_FORMAT_UNKNOWN;
  }
}

bool Check(HRESULT result, const char *operation) {
  if (SUCCEEDED(result))
    return true;
  ReportError(result, operation);
  return false;
}
}  // namespace grassland::rhi::backend::d3d12
