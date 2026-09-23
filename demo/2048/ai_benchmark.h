#pragma once

#include <cstdint>
#include <functional>
#include <vector>

// What one finished autoplay game reached.
struct AiBenchmarkGame {
  int max_number{0};
  int score{0};
  int moves{0};
  double average_depth{0.0};
  double average_nodes{0.0};
};

struct AiBenchmarkSettings {
  int games{1};
  uint32_t seed{1};
  float budget_ms{100.0f};
  std::function<void(const AiBenchmarkGame &game, int index)> on_finished;
};

// Plays the puzzle with the autoplay strategy on the rules of the game -
// update_step for the moves and the spawn rule of the application - without
// opening a window, so the strength of the strategy can be measured.
std::vector<AiBenchmarkGame> RunAiBenchmark(const AiBenchmarkSettings &settings);
