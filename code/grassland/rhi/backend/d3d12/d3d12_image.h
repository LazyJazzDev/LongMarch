#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12Image)
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"

namespace grassland::rhi::backend::d3d12 {

// Common resource access; derived classes define ownership and validity.
struct D3D12ImageState;

class ID3D12Image : public IBackendImage {
 public:
  static ObjectPtr<D3D12DescriptorPool::Lease> CreateAttachmentView(ID3D12Device *device, ID3D12Resource *image);
  D3D12_CPU_DESCRIPTOR_HANDLE AttachmentView() const;
  bool SupportsStorage() const override = 0;
  bool CommonSupportsStorage() const;
  const void *DeviceIdentity() const override = 0;
  const void *CommonDeviceIdentity() const;
  ID3D12Resource *NativeImage() const;
  D3D12_RESOURCE_STATES DefaultState() const;

 protected:
  void InitializeD3D12Image(ImageDesc desc,
                            const void *device,
                            ID3D12Resource *image,
                            ObjectPtr<D3D12DescriptorPool::Lease> attachment,
                            D3D12_RESOURCE_STATES default_state = D3D12_RESOURCE_STATE_COMMON,
                            IBackendQueue *queue = nullptr);
  virtual D3D12ImageState &StateD3D12Image() = 0;
  virtual const D3D12ImageState &StateD3D12Image() const = 0;
};

}  // namespace grassland::rhi::backend::d3d12
