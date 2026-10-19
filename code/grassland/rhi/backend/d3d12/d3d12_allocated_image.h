#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_image.h"

namespace D3D12MA {
class Allocation;
}

namespace grassland::rhi::backend::d3d12 {
class D3D12Device;

class D3D12AllocatedImage final : public D3D12Image {
 public:
  static D3D12AllocatedImage *Create(D3D12Device *device, const ImageDesc &desc);
  ~D3D12AllocatedImage() override;
  bool IsValid() const override;

 private:
  D3D12AllocatedImage(D3D12Device *device,
                      ImageDesc desc,
                      ComPtr<ID3D12Resource> image,
                      ComPtr<D3D12MA::Allocation> allocation);
  ComPtr<D3D12MA::Allocation> allocation_;
  ComPtr<ID3D12Resource> image_;
};
}  // namespace grassland::rhi::backend::d3d12
