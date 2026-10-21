#include "grassland/rhi/backend/d3d12/d3d12_image.h"

#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {

D3D12Image::D3D12Image(ImageDesc desc,
                       const void *device,
                       ID3D12Resource *image,
                       ObjectPtr<D3D12DescriptorPool::Lease> attachment,
                       D3D12_RESOURCE_STATES default_state)
    : Image(desc),
      attachment_(std::move(attachment)),
      device_(device),
      image_(image),
      default_state_(default_state) {
}

ObjectPtr<D3D12DescriptorPool::Lease> D3D12Image::CreateAttachmentView(D3D12Device *device, ID3D12Resource *image) {
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
  return attachment_->CPU();
}

bool D3D12Image::SupportsStorage() const {
  return (image_->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0;
}

const void *D3D12Image::DeviceIdentity() const {
  return device_;
}

ID3D12Resource *D3D12Image::NativeImage() const {
  return image_;
}

D3D12_RESOURCE_STATES D3D12Image::DefaultState() const {
  return default_state_;
}

}  // namespace grassland::rhi::backend::d3d12
