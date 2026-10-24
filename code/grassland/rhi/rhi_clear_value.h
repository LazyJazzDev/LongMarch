#pragma once
#include <array>
#include <cstdint>

namespace grassland::rhi {
union ClearColorValue {
  std::array<float, 4> float32;
  std::array<int32_t, 4> int32;
  std::array<uint32_t, 4> uint32;
};

struct ClearDepthStencilValue {
  float depth;
  uint32_t stencil;
};

// The image format selects the member: depth_stencil for depth formats,
// color.int32/uint32 for integer formats, color.float32 for all other colors.
// Initialize the matching member; no numeric conversion from float to integer
// is implied by the public interface.
union ClearValue {
  ClearColorValue color;
  ClearDepthStencilValue depth_stencil;

  ClearValue();
  ClearValue(float r, float g, float b, float a);
  ClearValue(const ClearColorValue &value);
  ClearValue(const ClearDepthStencilValue &value);
};
}  // namespace grassland::rhi
