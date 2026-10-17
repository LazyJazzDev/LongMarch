#include "grassland/rhi/backend/d3d12/d3d12_image.h"

namespace grassland::rhi::backend::d3d12 {

D3D12Image::D3D12Image(ImageDesc desc,
                       const void *device,
                       ID3D12Resource *image,
                       D3D12_CPU_DESCRIPTOR_HANDLE rtv,
                       D3D12_RESOURCE_STATES default_state)
    : Image(desc),
      device_(device),
      image_(image),
      rtv_(rtv),
      default_state_(default_state) {
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

D3D12_CPU_DESCRIPTOR_HANDLE D3D12Image::RenderTarget() const {
  return rtv_;
}

}  // namespace grassland::rhi::backend::d3d12
