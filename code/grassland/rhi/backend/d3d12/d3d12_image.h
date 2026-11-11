#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Image;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::D3D12Image)
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"

namespace grassland::rhi::backend::d3d12 {

class ID3D12Device;

// Common resource access; derived classes define ownership and validity.
struct D3D12ImageState;

class D3D12Image : public Image {
 public:
  static ObjectPtr<D3D12DescriptorPool::Lease> CreateAttachmentView(ID3D12Device *device, ID3D12Resource *image);
  D3D12_CPU_DESCRIPTOR_HANDLE AttachmentView() const;
  bool SupportsStorage() const override;
  const void *DeviceIdentity() const override;
  ID3D12Resource *NativeImage() const;
  D3D12_RESOURCE_STATES DefaultState() const;

 protected:
  void InitializeD3D12Image(ImageDesc desc,
                            const void *device,
                            ID3D12Resource *image,
                            ObjectPtr<D3D12DescriptorPool::Lease> attachment,
                            D3D12_RESOURCE_STATES default_state = D3D12_RESOURCE_STATE_COMMON,
                            Queue *queue = nullptr);
  virtual D3D12ImageState &StateD3D12Image() = 0;
  virtual const D3D12ImageState &StateD3D12Image() const = 0;
};

}  // namespace grassland::rhi::backend::d3d12
