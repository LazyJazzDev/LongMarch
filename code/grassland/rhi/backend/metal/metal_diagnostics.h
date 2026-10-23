#pragma once

#include <Metal/Metal.hpp>

namespace grassland::rhi::backend::metal {
void ReportMetalError(NS::Error *error, const char *operation, bool details = false);
}  // namespace grassland::rhi::backend::metal
