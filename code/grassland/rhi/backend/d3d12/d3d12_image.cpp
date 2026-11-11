#include "grassland/rhi/backend/d3d12/d3d12_image.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_image_state.h"

namespace grassland::rhi::backend::d3d12 {

void D3D12Image::InitializeD3D12Image(ImageDesc desc,
                                      const void *device,
                                      ID3D12Resource *image,
                                      ObjectPtr<D3D12DescriptorPool::Lease> attachment,
                                      D3D12_RESOURCE_STATES default_state,
                                      Queue *queue) {
  InitializeImage(desc, queue);
  StateD3D12Image().attachment_ = std::move(attachment);
  StateD3D12Image().device_ = device;
  StateD3D12Image().image_ = image;
  StateD3D12Image().default_state_ = default_state;
}

ObjectPtr<D3D12DescriptorPool::Lease> D3D12Image::CreateAttachmentView(ID3D12Device *device, ID3D12Resource *image) {
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

D3D12_CPU_DESCRIPTOR_HANDLE D3D12Image::AttachmentView() const {
  return StateD3D12Image().attachment_->CPU();
}

bool D3D12Image::SupportsStorage() const {
  return (StateD3D12Image().image_->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0;
}

const void *D3D12Image::DeviceIdentity() const {
  return StateD3D12Image().device_;
}

ID3D12Resource *D3D12Image::NativeImage() const {
  return StateD3D12Image().image_;
}

D3D12_RESOURCE_STATES D3D12Image::DefaultState() const {
  return StateD3D12Image().default_state_;
}

}  // namespace grassland::rhi::backend::d3d12

LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::D3D12Image, "85eb3c59-b643-4c32-b41f-66ed7e262a3e")

namespace grassland::rhi::backend::d3d12 {
D3D12ImageState::D3D12ImageState() = default;
D3D12ImageState::~D3D12ImageState() = default;
}  // namespace grassland::rhi::backend::d3d12
