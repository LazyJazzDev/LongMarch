#include "grassland/rhi/backend/d3d12/d3d12_image.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {

ObjectPtr<D3D12DescriptorPool::Lease> ID3D12Image::CreateAttachmentView(ID3D12Device *device, ID3D12Resource *image) {
  const bool depth = (image->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0;
  auto lease = (depth ? device->DSVPool() : device->RTVPool()).Acquire(1);
  if (!lease)
    return {};
  if (depth) {
    D3D12_DEPTH_STENCIL_VIEW_DESC view{};
    view.Format = image->GetDesc().Format == DXGI_FORMAT_R32_TYPELESS ? DXGI_FORMAT_D32_FLOAT : image->GetDesc().Format;
    view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    device->NativeDevice()->CreateDepthStencilView(image, &view, lease->CPU());
  } else
    device->NativeDevice()->CreateRenderTargetView(image, nullptr, lease->CPU());
  return lease;
}

}  // namespace grassland::rhi::backend::d3d12
