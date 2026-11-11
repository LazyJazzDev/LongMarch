#pragma once
#include <Metal/Metal.hpp>

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend::metal {
void ReportMetalError(NS::Error *error, const char *operation, bool details = false);
}  // namespace grassland::rhi::backend::metal
