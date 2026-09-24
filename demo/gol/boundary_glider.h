#pragma once

#include <algorithm>
#include <array>

#include "game_of_life_lib/game_of_life_lib.h"

// Evolve in empty space, then project onto the 4x4 icon. The padded board
// contains the complete 16-generation trip; projected cells never feed back.
class BoundaryGlider {
 public:
  static constexpr int kDisplaySize = 4;
  static constexpr int kGenerations = 16;
  static constexpr int kSpaceSize = 16;
  static constexpr int kOrigin = 4;

  BoundaryGlider() {
    Reset();
  }

  void Reset() {
    cells_.fill(0);
    for (auto offset : {1, kSpaceSize + 2, kSpaceSize * 2, kSpaceSize * 2 + 1, kSpaceSize * 2 + 2})
      cells_[kOrigin * kSpaceSize + kOrigin + offset] = 1;
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
    update_step(kSpaceSize, kSpaceSize, cells_.data(), BoundaryMode::kFixed);
    if (++generation_ == kGenerations)
      playing_ = false;
  }

  std::array<uint8_t, kDisplaySize * kDisplaySize> ProjectedCells() const {
    std::array<uint8_t, kDisplaySize * kDisplaySize> projected{};
    for (int y = 0; y < kSpaceSize; ++y)
      for (int x = 0; x < kSpaceSize; ++x)
        if (cells_[y * kSpaceSize + x])
          projected[(y % kDisplaySize) * kDisplaySize + x % kDisplaySize] = 1;
    return projected;
  }

  bool Playing() const {
    return playing_;
  }

  int Generation() const {
    return generation_;
  }

 private:
  std::array<uint8_t, kSpaceSize * kSpaceSize> cells_{};
  int generation_{};
  float elapsed_{};
  bool playing_{};
};
