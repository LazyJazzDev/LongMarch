#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12AllocatedImage)

namespace D3D12MA {
class Allocation;
}

namespace grassland::rhi::backend::d3d12 {
class ID3D12AllocatedImage : public ID3D12Image {
 public:
  static ID3D12AllocatedImage *Create(ID3D12Device *device, const ImageDesc &desc);
  virtual bool IsValid() const = 0;
};

;
}  // namespace grassland::rhi::backend::d3d12
