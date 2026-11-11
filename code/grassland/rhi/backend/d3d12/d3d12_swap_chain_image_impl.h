#pragma once
#include "grassland/rhi/backend/backend_image_state.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/d3d12/d3d12_image_state.h"
#include "grassland/rhi/backend/d3d12/d3d12_swap_chain_image.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12SwapChainImageImpl final : public Object<ID3D12SwapChainImage> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IImage, ID3D12SwapChainImage, ID3D12Image, IBackendImage, rhi::IImage)
  D3D12SwapChainImageImpl(ImageDesc desc,
                          ID3D12Window *window,
                          uint64_t generation,
                          ID3D12Resource *image,
                          ObjectPtr<D3D12DescriptorPool::Lease> attachment);
  ~D3D12SwapChainImageImpl() override;
  bool IsValid() const override;
  HRESULT Present();

 private:
  ID3D12Window *window_;
  uint64_t generation_;
#if !defined(NDEBUG)
  bool present_requested_{};
#endif

 private:
  D3D12ImageState state_D3D12Image_;

  D3D12ImageState &StateD3D12Image() override {
    return state_D3D12Image_;
  }

  const D3D12ImageState &StateD3D12Image() const override {
    return state_D3D12Image_;
  }

 private:
  ImageState state_Image_;

  ImageState &StateImage() override {
    return state_Image_;
  }

  const ImageState &StateImage() const override {
    return state_Image_;
  }

 public:
  bool SupportsStorage() const override {
    return ID3D12Image::CommonSupportsStorage();
  }

  const void *DeviceIdentity() const override {
    return ID3D12Image::CommonDeviceIdentity();
  }

  const ImageDesc &GetDesc() const override {
    return IBackendImage::CommonGetDesc();
  }
};
}  // namespace grassland::rhi::backend::d3d12
