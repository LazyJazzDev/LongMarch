#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_shader.h"

namespace grassland::rhi::backend {
IShader *CreateShader(const ShaderDesc &desc, BackendAPI api);

}  // namespace grassland::rhi::backend
