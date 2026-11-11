#pragma once
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_descriptor_pool.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::d3d12::ID3D12Image,
                        grassland::rhi::backend::IBackendImage,
                        "85eb3c59-b643-4c32-b41f-66ed7e262a3e")

namespace grassland::rhi::backend::d3d12 {

// Common resource access; derived classes define ownership and validity.
class ID3D12Image : public IBackendImage {
 public:
  static ObjectPtr<D3D12DescriptorPool::Lease> CreateAttachmentView(ID3D12Device *device, ID3D12Resource *image);
  virtual D3D12_CPU_DESCRIPTOR_HANDLE AttachmentView() const = 0;
  bool SupportsStorage() const override = 0;
  const void *DeviceIdentity() const override = 0;
  virtual ID3D12Resource *NativeImage() const = 0;
  virtual D3D12_RESOURCE_STATES DefaultState() const = 0;
};

}  // namespace grassland::rhi::backend::d3d12
