#pragma once

class SimulationClock {
 public:
  static constexpr int kLightning = 3;

  template <class Step, class Now>
  void Advance(double elapsed, bool playing, int speed, Step step, Now now) {
    if (speed != previous_speed_ || !playing)
      accumulated_ = 0.0;
    previous_speed_ = speed;
    if (!playing)
      return;
    if (speed == kLightning) {
      accumulated_ = 0.0;
      // Run continuously without a generation timer. Yield to input/rendering
      // after a short batch, rather than tying simulation speed to frame rate.
      const double deadline = now() + 0.008;
      do {
        step();
      } while (now() < deadline);
      return;
    }
    const double multiplier = speed == 1 ? 2.0 : speed == 2 ? 5.0 : 1.0;
    accumulated_ += elapsed * multiplier;
    while (accumulated_ >= 0.5) {
      step();
      accumulated_ -= 0.5;
    }
  }

 private:
  double accumulated_{0.0};
  int previous_speed_{0};
};
