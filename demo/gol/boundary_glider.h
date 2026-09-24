#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

// Recorded empty-space Life generations, projected modulo 4 for the icon.
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

  void Start() {
    Reset();
    playing_ = true;
  }

  void Update(float seconds) {
    if (!playing_)
      return;
    elapsed_ += std::max(seconds, 0.0f);
    // Let the wall open first. Advance at most once per rendered frame so a
    // slow frame cannot skip visible phases of the glider.
    if (elapsed_ < (generation_ == 0 ? 0.3f : 0.12f))
      return;
    elapsed_ = 0.0f;
    if (++generation_ == kGenerations)
      playing_ = false;
  }

  std::array<uint8_t, kDisplaySize * kDisplaySize> ProjectedCells() const {
    std::array<uint8_t, kDisplaySize * kDisplaySize> projected{};
    const uint16_t mask = kFrames[generation_ % kGenerations];
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
  inline static constexpr std::array<uint16_t, kGenerations> kFrames{0x2470, 0x0562, 0x0456, 0x02c6, 0x048e, 0x40ac,
                                                                     0xc08a, 0xc049, 0xd081, 0x9805, 0x5901, 0x3908,
                                                                     0x2b01, 0xa310, 0x2a30, 0x1630};
  int generation_{};
  float elapsed_{};
  bool playing_{};
};
