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

D3D12_DESCRIPTOR_RANGE_TYPE ResourceTypeToD3D12DescriptorRangeType(
    ResourceType type) {
  switch (type) {
    case RESOURCE_TYPE_UNIFORM_BUFFER:
      return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    case RESOURCE_TYPE_STORAGE_BUFFER:
      return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    case RESOURCE_TYPE_TEXTURE:
      return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    case RESOURCE_TYPE_IMAGE:
      return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    default:
      return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  }
}

D3D12_CULL_MODE CullModeToD3D12CullMode(CullMode mode) {
  switch (mode) {
    case CULL_MODE_NONE:
      return D3D12_CULL_MODE_NONE;
    case CULL_MODE_BACK:
      return D3D12_CULL_MODE_BACK;
    case CULL_MODE_FRONT:
      return D3D12_CULL_MODE_FRONT;
    default:
      return D3D12_CULL_MODE_NONE;
  }
}

D3D12ResourceBinding::D3D12ResourceBinding() : buffer(nullptr), image(nullptr) {
}

D3D12ResourceBinding::D3D12ResourceBinding(D3D12Buffer *buffer)
    : buffer(buffer), image(nullptr) {
}

D3D12ResourceBinding::D3D12ResourceBinding(D3D12Image *image)
    : buffer(nullptr), image(image) {
}

}  // namespace grassland::graphics::backend
