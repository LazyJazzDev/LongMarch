#pragma once
#include "grassland/graphics/backend/d3d12/helper/d3d12util.h"

namespace grassland::graphics::backend::d3d12 {
HRESULT SingleTimeCommand(ID3D12CommandQueue *queue,
                          ID3D12CommandAllocator *allocator,
                          const std::function<void(ID3D12GraphicsCommandList *)> &function);
}  // namespace grassland::graphics::backend::d3d12
