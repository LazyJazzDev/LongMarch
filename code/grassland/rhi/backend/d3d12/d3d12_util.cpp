#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
bool SupportsImageStorage(ID3D12Device *device, ImageFormat format) {
  const auto native = NativeImageFormat(format);
  if (native == DXGI_FORMAT_UNKNOWN || IsDepthFormat(format))
    return false;
  D3D12_FEATURE_DATA_FORMAT_SUPPORT support{native};
  if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))))
    return false;
  return (support.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) != 0;
}

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
