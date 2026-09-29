#pragma once
#include <algorithm>

class SimulationClock {
 public:
  static constexpr int kLightning = 3;
  // Lightning runs at a fixed 60 generations per second, independent of the
  // display refresh rate, so faster displays do not speed up the simulation.
  static constexpr double kLightningPeriod = 1.0 / 60.0;

  template <class Step>
  void Advance(double elapsed, bool playing, int speed, Step step) {
    if (speed != previous_speed_ || !playing) {
      // Entering lightning mode advances immediately, as before.
      accumulated_ = speed == kLightning ? kLightningPeriod : 0.0;
    } else {
      accumulated_ += speed == kLightning ? elapsed : 0.0;
    }
    previous_speed_ = speed;
    if (!playing)
      return;
    if (speed == kLightning) {
      // At most one generation per frame keeps motion even on slow frames. A
      // quarter-period tolerance and a bounded remainder lock the steps to a
      // 60 Hz display despite frame jitter: one per frame at 60 Hz, one every
      // other frame at 120 Hz.
      if (accumulated_ >= kLightningPeriod * 0.75) {
        step();
        accumulated_ = std::clamp(accumulated_ - kLightningPeriod, -kLightningPeriod * 0.25, kLightningPeriod * 0.25);
      }
      return;
    }
    const double multiplier = speed == 1 ? 2.0 : speed == 2 ? 5.0 : 1.0;
    accumulated_ += elapsed * multiplier;
    while (accumulated_ >= 0.5) {
      step();
      accumulated_ -= 0.5;
    }
  }

  double NextStepDelay(int speed) const {
    if (speed == kLightning)
      return std::max(0.0, kLightningPeriod * 0.75 - accumulated_);
    const double multiplier = speed == 1 ? 2.0 : speed == 2 ? 5.0 : 1.0;
    return std::max(0.0, (0.5 - accumulated_) / multiplier);
  }

 private:
  double accumulated_{0.0};
  int previous_speed_{0};
};
