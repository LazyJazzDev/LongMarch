#include "game_rules.h"

#include <vector>

std::optional<std::pair<int, int>> PickSpawnCell(std::mt19937 &random, const CellFlags &occupied, int *number) {
  std::vector<std::pair<int, int>> blank_positions;
  for (int x = 0; x < kBoardSize; x++) {
    for (int y = 0; y < kBoardSize; y++) {
      if (!occupied[BoardCell(x, y)]) {
        blank_positions.emplace_back(x, y);
      }
    }
  }

  if (blank_positions.empty()) {
    return std::nullopt;
  }

  const int pick = std::uniform_int_distribution<>(0, int(blank_positions.size()) - 1)(random);
  *number = std::uniform_real_distribution<>(0.0f, 1.0f)(random) < 0.9f ? 2 : 4;
  return blank_positions[pick];
}
