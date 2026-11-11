#pragma once
#include <d3d12sdklayers.h>

#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {

ICore *CreateD3D12Core(const CoreDesc &desc);

}  // namespace grassland::rhi::backend::d3d12
