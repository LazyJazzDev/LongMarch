#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12SwapChainImage;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12SwapChainImage)
#include "grassland/rhi/backend/d3d12/d3d12_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {

class ID3D12Window;

class ID3D12SwapChainImage : public D3D12Image {
 public:
  static ID3D12SwapChainImage *Create(ImageDesc desc,
                                      ID3D12Window *window,
                                      uint64_t generation,
                                      ID3D12Resource *image,
                                      ObjectPtr<D3D12DescriptorPool::Lease> attachment);
  virtual bool IsValid() const = 0;
  virtual HRESULT Present() = 0;
};

;

}  // namespace grassland::rhi::backend::d3d12
