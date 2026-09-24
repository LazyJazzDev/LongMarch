#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

// Recorded Life generations: empty-space modulo 4 or fixed 4x4 boundaries.
// Playback only selects a mask; it does not run the simulation.
class BoundaryGlider {
 public:
  static constexpr int kDisplaySize = 4;
  static constexpr int kGenerations = 4 * kDisplaySize;

  BoundaryGlider() {
    Reset();
  }

  void Reset() {
    generation_ = 0;
    elapsed_ = 0.0f;
    playing_ = false;
  }

  void Start(bool periodic = true) {
    Reset();
    periodic_ = periodic;
    playing_ = true;
  }

  void Update(float seconds) {
    if (!playing_)
      return;
    elapsed_ += std::max(seconds, 0.0f);
    // Let the wall open first. Advance at most once per rendered frame so a
    // slow frame cannot skip visible phases of the glider.
    const bool impact_hold = !periodic_ && generation_ == int(kFixedFrames.size()) - 1;
    if (elapsed_ < (impact_hold ? 0.7f : generation_ == 0 ? 0.3f : 0.12f))
      return;
    elapsed_ = 0.0f;
    if (impact_hold) {
      Reset();
      return;
    }
    if (++generation_ == kGenerations && periodic_)
      playing_ = false;
  }

  std::array<uint8_t, kDisplaySize * kDisplaySize> ProjectedCells() const {
    std::array<uint8_t, kDisplaySize * kDisplaySize> projected{};
    const uint16_t mask = periodic_ ? kFrames[generation_ % kGenerations] : kFixedFrames[generation_];
    for (int i = 0; i < kDisplaySize * kDisplaySize; ++i)
      projected[i] = (mask >> i) & 1;
    return projected;
  }

  bool Playing() const {
    return playing_;
  }

  int Generation() const {
    return generation_;
  }

 private:
  // Bit y*4+x: x increases rightward, y downward. The initial silhouette
  // occupies the lower-left 3x3 area; every four frames it moves (+1, -1).
  // Retain only even checkerboard tiles: floor(x/4)+floor(y/4) is even.
  // Single-edge crossings disappear; diagonal (two-edge) crossings reappear.
  inline static constexpr std::array<uint16_t, kGenerations> kFrames{0x2470, 0x0562, 0x0456, 0x02c6, 0x048e, 0x00ac,
                                                                     0x008a, 0x0048, 0x1080, 0x1004, 0x1100, 0x3108,
                                                                     0x2300, 0x2310, 0x2230, 0x1630};
  // True fixed-edge evolution ends in a stable block at the upper-right wall.
  inline static constexpr std::array<uint16_t, 8> kFixedFrames{0x2470, 0x0562, 0x0456, 0x02c6,
                                                               0x048e, 0x00ac, 0x008c, 0x00cc};
  bool periodic_{true};
  int generation_{};
  float elapsed_{};
  bool playing_{};
};
