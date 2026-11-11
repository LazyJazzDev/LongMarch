#pragma once
#include "grassland/rhi/backend/backend_image_state.h"
#include "grassland/rhi/backend/d3d12/d3d12_allocated_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_image_state.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12AllocatedImageImpl final : public Object<ID3D12AllocatedImage> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IImage, ID3D12AllocatedImage, D3D12Image, Image, rhi::IImage)
  static ID3D12AllocatedImage *Create(ID3D12Device *device, const ImageDesc &desc);
  ~D3D12AllocatedImageImpl() override;
  bool IsValid() const override;

 private:
  D3D12AllocatedImageImpl(ID3D12Device *device,
                          ImageDesc desc,
                          ComPtr<ID3D12Resource> image,
                          ComPtr<D3D12MA::Allocation> allocation,
                          ObjectPtr<D3D12DescriptorPool::Lease> attachment);
  ComPtr<D3D12MA::Allocation> allocation_;
  ComPtr<ID3D12Resource> image_;

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
};
}  // namespace grassland::rhi::backend::d3d12
