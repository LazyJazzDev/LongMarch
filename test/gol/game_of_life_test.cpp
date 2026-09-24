#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "../../demo/gol/game_of_life_lib/game_of_life_lib.h"

namespace {
std::vector<uint8_t> ReferenceStep(int width,
                                   int height,
                                   const std::vector<uint8_t> &cells,
                                   BoundaryMode mode = BoundaryMode::kPeriodic) {
  // Scatter each live cell into its eight periodic neighbors using a separate
  // count buffer. On a two-cell axis opposite directions intentionally coincide.
  std::vector<int> counts(cells.size());
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      if (cells[y * width + x])
        for (int dy : {-1, 0, 1})
          for (int dx : {-1, 0, 1})
            if ((dx || dy) &&
                (mode == BoundaryMode::kPeriodic || (x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height)))
              ++counts[((y + dy + height) % height) * width + (x + dx + width) % width];
  std::vector<uint8_t> next(cells.size());
  for (size_t i = 0; i < cells.size(); ++i)
    next[i] = counts[i] == 3 || (cells[i] && counts[i] == 2);
  return next;
}
}  // namespace

TEST(GameOfLife, ExhaustiveSmallGridsMatchSynchronousUpdates) {
  for (int width = 2; width <= 3; ++width) {
    for (int height = 2; height <= 3; ++height) {
      for (int pattern = 0; pattern < (1 << (width * height)); ++pattern) {
        std::vector<uint8_t> cells(width * height);
        for (size_t i = 0; i < cells.size(); ++i)
          cells[i] = (pattern >> i) & 1;
        for (int generation = 0; generation < 4; ++generation) {
          const auto expected = ReferenceStep(width, height, cells);
          update_step(width, height, cells.data());
          ASSERT_EQ(cells, expected) << width << 'x' << height << " pattern " << pattern;
        }
      }
    }
  }
}

TEST(GameOfLife, RepeatedUpdatesMatchAtMaximumAndThinDimensions) {
  std::mt19937 rng(42);
  for (int width : {2, 31, 200}) {
    for (int height : {2, 47, 200}) {
      std::vector<uint8_t> cells(width * height);
      for (auto &cell : cells)
        cell = rng() & 1;
      for (int generation = 0; generation < 20; ++generation) {
        const auto expected = ReferenceStep(width, height, cells);
        update_step(width, height, cells.data());
        ASSERT_EQ(cells, expected) << width << 'x' << height << " generation " << generation;
      }
    }
  }
}

TEST(GameOfLife, BlinkerOscillatesAcrossHorizontalSeam) {
  std::vector<uint8_t> cells(25);
  cells[2 * 5 + 4] = cells[2 * 5] = cells[2 * 5 + 1] = 1;
  const auto original = cells;
  std::vector<uint8_t> expected(25);
  expected[1 * 5] = expected[2 * 5] = expected[3 * 5] = 1;
  update_step(5, 5, cells.data());
  EXPECT_EQ(cells, expected);
  update_step(5, 5, cells.data());
  EXPECT_EQ(cells, original);
}

TEST(GameOfLife, BlinkerOscillatesAcrossVerticalSeam) {
  std::vector<uint8_t> cells(25);
  cells[4 * 5 + 2] = cells[2] = cells[1 * 5 + 2] = 1;
  const auto original = cells;
  std::vector<uint8_t> expected(25);
  expected[1] = expected[2] = expected[3] = 1;
  update_step(5, 5, cells.data());
  EXPECT_EQ(cells, expected);
  update_step(5, 5, cells.data());
  EXPECT_EQ(cells, original);
}

TEST(GameOfLife, CornerBlockIsStableAcrossBothSeams) {
  std::vector<uint8_t> cells(25);
  cells[0] = cells[4] = cells[20] = cells[24] = 1;
  const auto expected = cells;
  update_step(5, 5, cells.data());
  EXPECT_EQ(cells, expected);
}

TEST(GameOfLife, TwoCellAxesCountAllEightPeriodicOffsets) {
  // Three live neighbors by location contribute eight directional neighbors.
  std::vector<uint8_t> cells(4, 1);
  update_step(2, 2, cells.data());
  EXPECT_EQ(cells, std::vector<uint8_t>(4, 0));
}

TEST(GameOfLife, FixedBoundariesMatchReferenceAndCanSwitchWithoutReset) {
  std::mt19937 rng(73);
  for (int width : {2, 3, 31, 200}) {
    for (int height : {2, 3, 47, 200}) {
      std::vector<uint8_t> cells(width * height);
      for (auto &cell : cells)
        cell = rng() & 1;
      for (int generation = 0; generation < 20; ++generation) {
        const auto mode = generation % 3 == 0 ? BoundaryMode::kPeriodic : BoundaryMode::kFixed;
        const auto expected = ReferenceStep(width, height, cells, mode);
        update_step(width, height, cells.data(), mode);
        ASSERT_EQ(cells, expected) << width << 'x' << height << " generation " << generation;
      }
    }
  }
}

TEST(GameOfLife, FixedCornerCellsDieWhilePeriodicCornerBlockSurvives) {
  std::vector<uint8_t> periodic(25);
  periodic[0] = periodic[4] = periodic[20] = periodic[24] = 1;
  auto fixed = periodic;
  const auto original = periodic;
  update_step(5, 5, fixed.data(), BoundaryMode::kFixed);
  update_step(5, 5, periodic.data(), BoundaryMode::kPeriodic);
  EXPECT_EQ(fixed, std::vector<uint8_t>(25, 0));
  EXPECT_EQ(periodic, original);
}
