#include "grassland/rhi/rhi_clear_value.h"

namespace grassland::rhi {
ClearValue::ClearValue() : color{} {
}

ClearValue::ClearValue(float r, float g, float b, float a) : color{{r, g, b, a}} {
}

ClearValue::ClearValue(int32_t r, int32_t g, int32_t b, int32_t a) {
  color.int32 = {r, g, b, a};
}

ClearValue::ClearValue(uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
  color.uint32 = {r, g, b, a};
}

ClearValue::ClearValue(const ClearColorValue &value) : color(value) {
}

ClearValue::ClearValue(const ClearDepthStencilValue &value) : depth_stencil(value) {
}
}  // namespace grassland::rhi
