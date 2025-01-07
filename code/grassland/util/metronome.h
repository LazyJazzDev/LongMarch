#pragma once
#include <chrono>
#include <thread>

namespace grassland {

class Metronome {
 public:
  Metronome() : start_(std::chrono::steady_clock::now()) {
  }

  template <typename Duration>
  void Tick(const Duration &duration) {
    // Wait until the next tick
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<Duration>(now - start_);
    auto remaining = duration - elapsed;
    if (remaining.count() > 0) {
      std::this_thread::sleep_for(remaining);
    }
    start_ = std::chrono::steady_clock::now();
  }

 private:
  std::chrono::steady_clock::time_point start_;
};

}  // namespace grassland
