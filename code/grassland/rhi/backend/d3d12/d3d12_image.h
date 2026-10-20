#pragma once
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {

// Common resource access; derived classes define ownership and validity.
class D3D12Image : public Image {
 public:
  bool SupportsStorage() const override;
  const void *DeviceIdentity() const override;
  ID3D12Resource *NativeImage() const;
  D3D12_RESOURCE_STATES DefaultState() const;

 protected:
  D3D12Image(ImageDesc desc,
             const void *device,
             ID3D12Resource *image,
             D3D12_RESOURCE_STATES default_state = D3D12_RESOURCE_STATE_COMMON);

 private:
  const void *device_;
  ID3D12Resource *image_;
  D3D12_RESOURCE_STATES default_state_;
};

}  // namespace grassland::rhi::backend::d3d12
