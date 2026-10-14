#include "grassland/rhi/backend/d3d12/d3d12_image.h"

#include "grassland/rhi/backend/d3d12/d3d12_window.h"

namespace grassland::rhi::backend::d3d12 {

D3D12Image::D3D12Image(ImageDesc desc,
                       const ObjectPtr<D3D12Window> &window,
                       uint64_t generation,
                       ID3D12Resource *image,
                       D3D12_CPU_DESCRIPTOR_HANDLE rtv)
    : IImage(desc),
      window_(window),
      generation_(generation),
      image_(image),
      rtv_(rtv) {
}

D3D12Image::~D3D12Image() {
  if (!presented_)
    window_->CancelFrame(generation_);
}

bool D3D12Image::IsValid() const {
  return !presented_ && window_->IsFrameValid(generation_);
}

const void *D3D12Image::DeviceIdentity() const {
  return window_->DeviceIdentity();
}

ID3D12Resource *D3D12Image::NativeImage() const {
  return image_;
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12Image::RenderTarget() const {
  return rtv_;
}

HRESULT D3D12Image::Present() {
  presented_ = true;
  return window_->PresentFrame(generation_);
}

}  // namespace grassland::rhi::backend::d3d12
