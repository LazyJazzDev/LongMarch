#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12SwapChainImage;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::D3D12SwapChainImage)
#include "grassland/rhi/backend/d3d12/d3d12_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {

class D3D12Window;

class D3D12SwapChainImage final : public D3D12Image {
 public:
  LONGMARCH_RHI_QUERY(D3D12SwapChainImage, D3D12Image)

  D3D12SwapChainImage(ImageDesc desc,
                      D3D12Window *window,
                      uint64_t generation,
                      ID3D12Resource *image,
                      ObjectPtr<D3D12DescriptorPool::Lease> attachment);
  ~D3D12SwapChainImage() override;
  bool IsValid() const override;
  HRESULT Present();

 private:
  D3D12Window *window_;
  uint64_t generation_;
#if !defined(NDEBUG)
  bool present_requested_{};
#endif
};

}  // namespace grassland::rhi::backend::d3d12
