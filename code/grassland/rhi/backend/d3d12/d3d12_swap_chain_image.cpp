#include "grassland/rhi/backend/d3d12/d3d12_swap_chain_image.h"

#include "grassland/rhi/backend/d3d12/d3d12_window.h"

namespace grassland::rhi::backend::d3d12 {

D3D12SwapChainImage::D3D12SwapChainImage(ImageDesc desc,
                                         D3D12Window *window,
                                         uint64_t generation,
                                         ID3D12Resource *image,
                                         D3D12_CPU_DESCRIPTOR_HANDLE rtv)
    : D3D12Image(desc, window->DeviceIdentity(), image, rtv),
      window_(window),
      generation_(generation) {
}

D3D12SwapChainImage::~D3D12SwapChainImage() {
  if (!presented_)
    window_->CancelFrame(generation_);
}

const D3D12Window *D3D12SwapChainImage::Owner() const {
  return window_;
}

bool D3D12SwapChainImage::IsValid() const {
  return !presented_ && window_->IsFrameValid(generation_);
}

HRESULT D3D12SwapChainImage::Present() {
  presented_ = true;
  return window_->PresentFrame(generation_);
}

}  // namespace grassland::rhi::backend::d3d12
