#pragma once
#include <unordered_map>

#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/metal/metal_acceleration_structure.h"
#include "grassland/rhi/backend/metal/metal_binding_group.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
ICommandContext *CreateMetalCommandContext(IMetalQueue *queue);

}  // namespace grassland::rhi::backend::metal
