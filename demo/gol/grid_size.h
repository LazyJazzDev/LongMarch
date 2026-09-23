#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace grid_size {
inline constexpr int kMin = 2;
inline constexpr int kMax = 200;

inline int FromFraction(float fraction) {
  return kMin + int(std::lround(std::clamp(fraction, 0.0f, 1.0f) * (kMax - kMin)));
}

inline std::vector<uint8_t> Resize(const std::vector<uint8_t> &cells,
                                   int old_width,
                                   int old_height,
                                   int width,
                                   int height) {
  std::vector<uint8_t> resized(width * height, 0);
  for (int y = 0; y < std::min(old_height, height); ++y)
    std::copy_n(cells.begin() + y * old_width, std::min(old_width, width), resized.begin() + y * width);
  return resized;
}
}  // namespace grid_size
