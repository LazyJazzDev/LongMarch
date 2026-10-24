#include "grassland/rhi/backend/d3d12/d3d12_util.h"

#include "grassland/rhi/backend/backend_binding_group.h"

namespace grassland::rhi::backend::d3d12 {
D3D12_BLEND NativeBlendFactor(BlendFactor factor, bool alpha) {
  // Alpha equations use the alpha component of color factors as well.
  if (alpha && factor >= BLEND_FACTOR_SRC_COLOR && factor <= BLEND_FACTOR_ONE_MINUS_DST_COLOR)
    factor = static_cast<BlendFactor>(factor + BLEND_FACTOR_SRC_ALPHA - BLEND_FACTOR_SRC_COLOR);
  const D3D12_BLEND factors[] = {D3D12_BLEND_ZERO,          D3D12_BLEND_ONE,           D3D12_BLEND_SRC_COLOR,
                                 D3D12_BLEND_INV_SRC_COLOR, D3D12_BLEND_DEST_COLOR,    D3D12_BLEND_INV_DEST_COLOR,
                                 D3D12_BLEND_SRC_ALPHA,     D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_DEST_ALPHA,
                                 D3D12_BLEND_INV_DEST_ALPHA};
  return factors[factor];
}

D3D12_BLEND_OP NativeBlendOp(BlendOp op) {
  const D3D12_BLEND_OP operations[] = {D3D12_BLEND_OP_ADD, D3D12_BLEND_OP_SUBTRACT, D3D12_BLEND_OP_REV_SUBTRACT,
                                       D3D12_BLEND_OP_MIN, D3D12_BLEND_OP_MAX};
  return operations[op];
}

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
    case IMAGE_FORMAT_R8_UINT:
      return DXGI_FORMAT_R8_UINT;
    case IMAGE_FORMAT_R8_INT:
      return DXGI_FORMAT_R8_SINT;
    case IMAGE_FORMAT_R8_UNORM:
      return DXGI_FORMAT_R8_UNORM;
    case IMAGE_FORMAT_R8G8_UINT:
      return DXGI_FORMAT_R8G8_UINT;
    case IMAGE_FORMAT_R8G8_INT:
      return DXGI_FORMAT_R8G8_SINT;
    case IMAGE_FORMAT_R8G8_UNORM:
      return DXGI_FORMAT_R8G8_UNORM;
    case IMAGE_FORMAT_R8G8B8A8_UINT:
      return DXGI_FORMAT_R8G8B8A8_UINT;
    case IMAGE_FORMAT_R8G8B8A8_INT:
      return DXGI_FORMAT_R8G8B8A8_SINT;
    case IMAGE_FORMAT_R8G8B8A8_UNORM:
      return DXGI_FORMAT_R8G8B8A8_UNORM;
    case IMAGE_FORMAT_R32_UINT:
      return DXGI_FORMAT_R32_UINT;
    case IMAGE_FORMAT_R32_INT:
      return DXGI_FORMAT_R32_SINT;
    case IMAGE_FORMAT_R32_SFLOAT:
      return DXGI_FORMAT_R32_FLOAT;
    case IMAGE_FORMAT_R32G32_UINT:
      return DXGI_FORMAT_R32G32_UINT;
    case IMAGE_FORMAT_R32G32_INT:
      return DXGI_FORMAT_R32G32_SINT;
    case IMAGE_FORMAT_R32G32_SFLOAT:
      return DXGI_FORMAT_R32G32_FLOAT;
    case IMAGE_FORMAT_R32G32B32_UINT:
      return DXGI_FORMAT_R32G32B32_UINT;
    case IMAGE_FORMAT_R32G32B32_INT:
      return DXGI_FORMAT_R32G32B32_SINT;
    case IMAGE_FORMAT_R32G32B32_SFLOAT:
      return DXGI_FORMAT_R32G32B32_FLOAT;
    case IMAGE_FORMAT_R32G32B32A32_UINT:
      return DXGI_FORMAT_R32G32B32A32_UINT;
    case IMAGE_FORMAT_R32G32B32A32_INT:
      return DXGI_FORMAT_R32G32B32A32_SINT;
    case IMAGE_FORMAT_R32G32B32A32_SFLOAT:
      return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case IMAGE_FORMAT_B8G8R8A8_UNORM:
      return DXGI_FORMAT_B8G8R8A8_UNORM;
    case IMAGE_FORMAT_R16G16B16A16_SFLOAT:
      return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case IMAGE_FORMAT_R10G10B10A2_UNORM:
      return DXGI_FORMAT_R10G10B10A2_UNORM;
    case IMAGE_FORMAT_D16_UNORM:
      return DXGI_FORMAT_D16_UNORM;
    case IMAGE_FORMAT_D32_SFLOAT:
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
