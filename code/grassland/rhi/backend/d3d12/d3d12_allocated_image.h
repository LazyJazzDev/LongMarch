#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace D3D12MA {
class Allocation;
}

namespace grassland::rhi::backend::d3d12 {
IImage *CreateD3D12AllocatedImage(ID3D12Device *device, const ImageDesc &desc);

}  // namespace grassland::rhi::backend::d3d12
