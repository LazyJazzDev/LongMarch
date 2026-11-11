#pragma once
#include <vector>

#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/d3d12/d3d12_binding_group.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
ICommandContext *CreateD3D12CommandContext(ID3D12Queue *queue);

}  // namespace grassland::rhi::backend::d3d12
