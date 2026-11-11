#include "grassland/rhi/backend/d3d12/d3d12_image.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {

ObjectPtr<D3D12DescriptorPool::Lease> ID3D12Image::CreateAttachmentView(ID3D12Device *device, ID3D12Resource *image) {
  const bool depth = (image->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0;
  auto lease = (depth ? device->DSVPool() : device->RTVPool()).Acquire(1);
  if (!lease)
    return {};
  if (depth)
    device->NativeDevice()->CreateDepthStencilView(image, nullptr, lease->CPU());
  else
    device->NativeDevice()->CreateRenderTargetView(image, nullptr, lease->CPU());
  return lease;
}

}  // namespace grassland::rhi::backend::d3d12

LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::ID3D12Image, "85eb3c59-b643-4c32-b41f-66ed7e262a3e")
