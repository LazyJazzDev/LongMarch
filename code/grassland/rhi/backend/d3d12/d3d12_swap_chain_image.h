#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::ID3D12SwapChainImage, "f03d8d79-9e19-41ca-b585-8f6f5bd64143")

namespace grassland::rhi::backend::d3d12 {

class ID3D12SwapChainImage : public ID3D12Image {
 public:
  static ID3D12SwapChainImage *Create(ImageDesc desc,
                                      ID3D12Window *window,
                                      uint64_t generation,
                                      ID3D12Resource *image,
                                      ObjectPtr<D3D12DescriptorPool::Lease> attachment);
  virtual HRESULT Present() = 0;
};

}  // namespace grassland::rhi::backend::d3d12
