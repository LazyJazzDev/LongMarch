#pragma once

#include <chrono>

namespace grassland {

// Monotonic seconds since the first call; independent of window initialization.
inline double GetTimeSeconds() {
  static const auto epoch = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch).count();
}

}  // namespace grassland
