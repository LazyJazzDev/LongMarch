#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {

class D3D12Window;

class D3D12SwapChainImage final : public D3D12Image {
 public:
  D3D12SwapChainImage(ImageDesc desc,
                      D3D12Window *window,
                      uint64_t generation,
                      ID3D12Resource *image,
                      D3D12_CPU_DESCRIPTOR_HANDLE rtv);
  ~D3D12SwapChainImage() override;
  bool IsValid() const override;
  HRESULT Present();

 private:
  D3D12Window *window_;
  uint64_t generation_;
  std::atomic<bool> presented_{};
};

}  // namespace grassland::rhi::backend::d3d12
