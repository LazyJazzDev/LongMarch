#pragma once
#include <algorithm>

class SimulationClock {
 public:
  static constexpr int kLightning = 3;

  template <class Step>
  void Advance(double elapsed, bool playing, int speed, Step step) {
    if (speed != previous_speed_ || !playing)
      accumulated_ = 0.0;
    previous_speed_ = speed;
    if (!playing)
      return;
    if (speed == kLightning) {
      accumulated_ = 0.0;
      // One generation per rendered frame, without an additional timer.
      step();
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
      return 0;
    const double multiplier = speed == 1 ? 2.0 : speed == 2 ? 5.0 : 1.0;
    return std::max(0.0, (0.5 - accumulated_) / multiplier);
  }

 private:
  double accumulated_{0.0};
  int previous_speed_{0};
};
