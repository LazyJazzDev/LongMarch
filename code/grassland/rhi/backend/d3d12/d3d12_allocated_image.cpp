#include "grassland/rhi/backend/d3d12/d3d12_allocated_image.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#define D3D12MA_D3D12_HEADERS_ALREADY_INCLUDED
#include <D3D12MemAlloc.h>
#undef D3D12MA_D3D12_HEADERS_ALREADY_INCLUDED

namespace grassland::rhi::backend::d3d12 {
D3D12AllocatedImage::D3D12AllocatedImage(D3D12Device *device,
                                         ImageDesc desc,
                                         ComPtr<ID3D12Resource> image,
                                         ComPtr<D3D12MA::Allocation> allocation)
    : D3D12Image(desc, device, image.Get()),
      allocation_(std::move(allocation)),
      image_(std::move(image)) {
}

D3D12AllocatedImage::~D3D12AllocatedImage() = default;

bool D3D12AllocatedImage::IsValid() const {
  return true;
}

D3D12AllocatedImage *D3D12AllocatedImage::Create(D3D12Device *device, const ImageDesc &desc) {
  if (!desc.width || !desc.height || (desc.storage && desc.format != IMAGE_FORMAT_RGBA8_UNORM))
    return nullptr;
  DXGI_FORMAT format;
  switch (desc.format) {
    case IMAGE_FORMAT_RGBA8_UNORM:
      format = DXGI_FORMAT_R8G8B8A8_UNORM;
      break;
    case IMAGE_FORMAT_BGRA8_UNORM:
      format = DXGI_FORMAT_B8G8R8A8_UNORM;
      break;
    case IMAGE_FORMAT_D32_FLOAT:
      format = DXGI_FORMAT_D32_FLOAT;
      break;
    case IMAGE_FORMAT_D24_UNORM_S8_UINT:
      format = DXGI_FORMAT_D24_UNORM_S8_UINT;
      break;
    default:
      return nullptr;
  }
  try {
    D3D12_RESOURCE_DESC info{};
    info.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    info.Width = desc.width;
    info.Height = desc.height;
    info.DepthOrArraySize = 1;
    info.MipLevels = 1;
    info.Format = format;
    info.SampleDesc.Count = 1;
    info.Flags =
        IsDepthFormat(desc.format) ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL : D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if (desc.storage)
      info.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12MA::ALLOCATION_DESC memory{};
    memory.HeapType = D3D12_HEAP_TYPE_DEFAULT;
    ComPtr<ID3D12Resource> image;
    ComPtr<D3D12MA::Allocation> allocation;
    if (!Check(device->Allocator()->CreateResource(&memory, &info, D3D12_RESOURCE_STATE_COMMON, nullptr, &allocation,
                                                   IID_PPV_ARGS(&image)),
               "Allocate image"))
      return nullptr;
    return new D3D12AllocatedImage(device, desc, std::move(image), std::move(allocation));
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Image allocation ran out of memory.");
    return nullptr;
  }
}
}  // namespace grassland::rhi::backend::d3d12
