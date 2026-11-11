#include "grassland/rhi/backend/d3d12/d3d12_swap_chain_image.h"

#include "grassland/rhi/backend/d3d12/d3d12_swap_chain_image_impl.h"
#include "grassland/rhi/backend/d3d12/d3d12_window.h"

namespace grassland::rhi::backend::d3d12 {

D3D12SwapChainImageImpl::D3D12SwapChainImageImpl(ImageDesc desc,
                                                 ID3D12Window *window,
                                                 uint64_t generation,
                                                 ID3D12Resource *image,
                                                 ObjectPtr<D3D12DescriptorPool::Lease> attachment)
    : window_(window),
      generation_(generation) {
  InitializeD3D12Image(desc, window->DeviceIdentity(), image, std::move(attachment), D3D12_RESOURCE_STATE_COMMON,
                       window->OwnerQueue());
}

D3D12SwapChainImageImpl::~D3D12SwapChainImageImpl() {
}

bool D3D12SwapChainImageImpl::IsValid() const {
#if !defined(NDEBUG)
  if (present_requested_)
    return false;
#endif
  return window_->IsFrameValid(generation_);
}

HRESULT D3D12SwapChainImageImpl::Present() {
#if !defined(NDEBUG)
  present_requested_ = true;
#endif
  return window_->PresentFrame(generation_);
}

}  // namespace grassland::rhi::backend::d3d12

LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::ID3D12SwapChainImage, "f03d8d79-9e19-41ca-b585-8f6f5bd64143")

namespace grassland::rhi::backend::d3d12 {
ID3D12SwapChainImage *ID3D12SwapChainImage::Create(ImageDesc desc,
                                                   ID3D12Window *window,
                                                   uint64_t generation,
                                                   ID3D12Resource *image,
                                                   ObjectPtr<D3D12DescriptorPool::Lease> attachment) {
  return new D3D12SwapChainImageImpl(desc, window, generation, image, attachment);
}
}  // namespace grassland::rhi::backend::d3d12
