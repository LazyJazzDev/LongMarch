#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_timeline.h"
#include "grassland/rhi/rhi_cuda_timeline.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
namespace grassland::rhi::backend::d3d12 {
ObjectPtr<ID3D12Timeline> CreateD3D12CUDATimeline(ID3D12Device *device);
}
#endif
