#include "grassland/d3d12/d3d12util.h"

namespace grassland::d3d12 {

std::string HRESULTToString(HRESULT hr) {
  char buffer[256] = {};
  sprintf(buffer, "HRESULT: 0x%08X", hr);
  return buffer;
}

size_t SizeByFormat(DXGI_FORMAT format) {
  switch (format) {
    case DXGI_FORMAT_R8_SINT:
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8_SNORM:
      return 1;

    case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8G8_SNORM:
      return 2;

    case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_R8G8B8A8_UINT:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_SNORM:
      return 4;

    case DXGI_FORMAT_R32_SINT:
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT:
      return 4;

    case DXGI_FORMAT_R32G32_SINT:
    case DXGI_FORMAT_R32G32_UINT:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32G32_TYPELESS:
      return 8;

    case DXGI_FORMAT_R32G32B32_SINT:
    case DXGI_FORMAT_R32G32B32_UINT:
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32B32_TYPELESS:
      return 12;

    case DXGI_FORMAT_R32G32B32A32_SINT:
    case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
      return 16;

    default:
      return 0;
  }
}

size_t SizeAlignTo(size_t size, size_t alignment) {
  return (size + alignment - 1) & ~(alignment - 1);
}

bool IsDepthFormat(DXGI_FORMAT format) {
  switch (format) {
    case DXGI_FORMAT_D16_UNORM:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_D32_FLOAT:
      return true;
    default:
      return false;
  }
}

HRESULT CreateBuffer(ID3D12Device *device,
                     size_t size,
                     D3D12_HEAP_TYPE heap_type,
                     D3D12_HEAP_FLAGS heap_flags,
                     D3D12_RESOURCE_STATES resource_state,
                     D3D12_RESOURCE_FLAGS resource_flags,
                     ComPtr<ID3D12Resource> &buffer) {
  auto heap_properties = CD3DX12_HEAP_PROPERTIES(heap_type);
  auto resource_desc = CD3DX12_RESOURCE_DESC::Buffer(SizeAlignTo(size, 256), resource_flags);
  RETURN_IF_FAILED_HR(device->CreateCommittedResource(&heap_properties, heap_flags, &resource_desc, resource_state,
                                                      nullptr, IID_PPV_ARGS(&buffer)),
                      "failed to create buffer.");
  return S_OK;
}

HRESULT CreateBuffer(ID3D12Device *device,
                     size_t size,
                     D3D12_HEAP_TYPE heap_type,
                     D3D12_RESOURCE_STATES resource_state,
                     D3D12_RESOURCE_FLAGS resource_flags,
                     ComPtr<ID3D12Resource> &buffer) {
  return CreateBuffer(device, size, heap_type, D3D12_HEAP_FLAG_NONE, resource_state, resource_flags, buffer);
}

}  // namespace grassland::d3d12
