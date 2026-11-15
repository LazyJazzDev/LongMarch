#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

namespace life_demo {
// Shared CPU policy for checker transitions and the coordinate ruler.
struct GridDetail {
  static constexpr float kContrast = 0.15f;
  static constexpr std::array<int, 4> kSpans{2048, 256, 32, 4};
  int major{};
  float transition{};

  static GridDetail At(float pitch, float short_edge) {
    GridDetail result;
    while (result.major < 3 && kSpans[result.major] * pitch >= short_edge)
      ++result.major;
    if (result.major < 3)
      result.transition =
          std::clamp(2.0f * kSpans[result.major] * pitch / std::max(short_edge, 1.0f) - 1.0f, 0.0f, 1.0f);
    return result;
  }

  uint32_t Packed() const {
    return (uint32_t(major) << 16) | uint32_t(transition * 65535.0f + 0.5f);
  }

  static bool Heat(float pitch) {
    return pitch < 0.25f;
  }
};

struct GridRulerTiming {
  double started{-10}, last{-10};

  void Zoom(double now) {
    if (now >= last + 2.35)
      started = now;
    last = now;
  }

  float Opacity(double now) const {
    return float(std::clamp((now - started) / 0.15, 0.0, 1.0) * std::clamp((last + 2.35 - now) / 0.35, 0.0, 1.0));
  }

  double NextDelay(double now, double otherwise) const {
    if (now < started + 0.15 || (now >= last + 2 && now < last + 2.35))
      return 0;
    if (now < last + 2)
      return std::min(otherwise, last + 2 - now);
    return otherwise;
  }
};
}  // namespace life_demo
