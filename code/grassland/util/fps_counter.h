#pragma once
#include <chrono>
#include <queue>

namespace grassland {
class FPSCounter {
 public:
  void TickFrame();
  float GetFPS() const;
  float TickFPS();
  void Reset();

 private:
  std::queue<std::chrono::high_resolution_clock::time_point> frames_;
};
}  // namespace grassland
