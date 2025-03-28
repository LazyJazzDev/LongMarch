#pragma once
#include <chrono>
#include <queue>

#include "grassland/util/util_util.h"

namespace grassland {
class FPSCounter {
 public:
  void TickFrame();
  float GetFPS() const;
  float TickFPS();
  void Reset();

  static void PyBind(pybind11::module_ &m);

 private:
  std::queue<std::chrono::high_resolution_clock::time_point> frames_;
};
}  // namespace grassland
