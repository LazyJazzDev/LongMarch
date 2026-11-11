#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12AllocatedImage)
#include "grassland/rhi/backend/d3d12/d3d12_image.h"

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
