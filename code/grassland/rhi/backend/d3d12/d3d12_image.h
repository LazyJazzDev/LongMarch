#pragma once
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {

// Common resource access; derived classes define ownership and validity.
class D3D12Image : public Image {
 public:
  const void *DeviceIdentity() const override;
  ID3D12Resource *NativeImage() const;
  D3D12_CPU_DESCRIPTOR_HANDLE RenderTarget() const;

 protected:
  D3D12Image(ImageDesc desc, const void *device, ID3D12Resource *image, D3D12_CPU_DESCRIPTOR_HANDLE rtv);

 private:
  const void *device_;
  ID3D12Resource *image_;
  D3D12_CPU_DESCRIPTOR_HANDLE rtv_;
};

}  // namespace grassland::rhi::backend::d3d12
