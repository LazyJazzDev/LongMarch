#pragma once
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"

namespace grassland::rhi::backend::d3d12 {

class D3D12Device;

// Common resource access; derived classes define ownership and validity.
class D3D12Image : public Image {
 public:
  static ObjectPtr<D3D12DescriptorPool::Lease> CreateAttachmentView(D3D12Device *device, ID3D12Resource *image);
  D3D12_CPU_DESCRIPTOR_HANDLE AttachmentView() const;
  bool SupportsStorage() const override;
  const void *DeviceIdentity() const override;
  ID3D12Resource *NativeImage() const;
  D3D12_RESOURCE_STATES DefaultState() const;

 protected:
  D3D12Image(ImageDesc desc,
             const void *device,
             ID3D12Resource *image,
             ObjectPtr<D3D12DescriptorPool::Lease> attachment,
             D3D12_RESOURCE_STATES default_state = D3D12_RESOURCE_STATE_COMMON);

 private:
  ObjectPtr<D3D12DescriptorPool::Lease> attachment_;
  const void *device_;
  ID3D12Resource *image_;
  D3D12_RESOURCE_STATES default_state_;
};

}  // namespace grassland::rhi::backend::d3d12
