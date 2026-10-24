#include "grassland/rhi/rhi_clear_value.h"

namespace grassland::rhi {
ClearValue::ClearValue() : color{} {
}

ClearValue::ClearValue(float r, float g, float b, float a) : color{{r, g, b, a}} {
}

ClearValue::ClearValue(const ClearColorValue &value) : color(value) {
}

ClearValue::ClearValue(const ClearDepthStencilValue &value) : depth_stencil(value) {
}
}  // namespace grassland::rhi
