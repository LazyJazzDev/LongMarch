#include "../../demo/gol/cells_pattern.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "../../demo/gol/game_of_life_lib/game_of_life_lib.h"

TEST(GameOfLife, Lexicon295P5H1V1MovesOneCellDiagonallyEveryFiveGenerations) {
  const auto pattern = LoadCellsPattern(GOL_PATTERN_FILE);
  ASSERT_EQ(pattern.width, 52);
  ASSERT_EQ(pattern.height, 52);
  ASSERT_EQ(std::count(pattern.cells.begin(), pattern.cells.end(), 1), 295);

  constexpr int kGridSize = 200;
  constexpr int kOffset = (kGridSize - 52) / 2;
  auto cells = CenterCellsPattern(pattern, kGridSize, kGridSize);
  for (int generation = 0; generation < 5; ++generation)
    update_step(kGridSize, kGridSize, cells.data());

  std::vector<uint8_t> expected(kGridSize * kGridSize, 0);
  for (int y = 0; y < pattern.height; ++y)
    for (int x = 0; x < pattern.width; ++x)
      expected[(kOffset + y - 1) * kGridSize + kOffset + x - 1] = pattern.cells[y * pattern.width + x];
  EXPECT_EQ(cells, expected);
}
