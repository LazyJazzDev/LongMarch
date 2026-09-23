#include "../../demo/gol/grid_size.h"

#include <gtest/gtest.h>

TEST(GridSize, SliderEndpointsAndIntegerSteps) {
  EXPECT_EQ(grid_size::FromFraction(-1), 2);
  EXPECT_EQ(grid_size::FromFraction(0), 2);
  EXPECT_EQ(grid_size::FromFraction(1), 200);
  EXPECT_EQ(grid_size::FromFraction(2), 200);
  for (int size = 2; size <= 200; ++size)
    EXPECT_EQ(grid_size::FromFraction(float(size - 2) / 198.0f), size);
}

TEST(GridSize, ResizePreservesCoordinatesAndClearsNewCells) {
  std::vector<uint8_t> cells{1, 0, 1, 0, 1, 0};
  auto expanded = grid_size::Resize(cells, 3, 2, 5, 4);
  EXPECT_EQ(expanded, (std::vector<uint8_t>{1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  auto shrunk = grid_size::Resize(expanded, 5, 4, 2, 2);
  EXPECT_EQ(shrunk, (std::vector<uint8_t>{1, 0, 0, 1}));
  auto maximum = grid_size::Resize(shrunk, 2, 2, 200, 200);
  EXPECT_EQ(maximum.size(), 40000);
  EXPECT_EQ(maximum[0], 1);
  EXPECT_EQ(maximum[201], 1);
  EXPECT_EQ(std::count(maximum.begin(), maximum.end(), 1), 2);
}
