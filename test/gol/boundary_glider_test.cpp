#include "../../demo/gol/boundary_glider.h"

#include <gtest/gtest.h>

TEST(BoundaryGlider, EvolvesInEmptySpaceBeforeProjectingAndStopsAfterOneTrip) {
  BoundaryGlider glider;
  const auto initial = glider.ProjectedCells();
  glider.Start();
  glider.Update(0.29f);
  EXPECT_EQ(glider.Generation(), 0);
  glider.Update(0.02f);
  // Evolve in empty space, then wrap the first generation onto the 3x3 display.
  std::array<uint8_t, 9> first{};
  for (int index : {3, 5, 7, 8, 1})
    first[index] = 1;
  EXPECT_EQ(glider.ProjectedCells(), first);
  for (int generation = 2; generation <= 12; ++generation) {
    glider.Update(0.13f);
    const auto cells = glider.ProjectedCells();
    EXPECT_EQ(std::count(cells.begin(), cells.end(), 1), 5);
    EXPECT_EQ(glider.Generation(), generation);
    EXPECT_EQ(glider.Playing(), generation < 12);
    if (generation % 4 == 0) {
      std::array<uint8_t, 9> translated{};
      const int shift = generation / 4;
      for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 3; ++x)
          translated[((y + shift) % 3) * 3 + (x + shift) % 3] = initial[y * 3 + x];
      EXPECT_EQ(cells, translated);
    }
  }
  EXPECT_EQ(glider.ProjectedCells(), initial);
  glider.Update(100.0f);
  EXPECT_EQ(glider.Generation(), 12);
  EXPECT_EQ(glider.ProjectedCells(), initial);
}

TEST(BoundaryGlider, ResetAndRestartDoNotRetainAnimationState) {
  BoundaryGlider glider;
  const auto initial = glider.ProjectedCells();
  glider.Start();
  glider.Update(5.0f);
  EXPECT_EQ(glider.Generation(), 1);  // Never skip visible generations.
  glider.Reset();
  EXPECT_FALSE(glider.Playing());
  EXPECT_EQ(glider.ProjectedCells(), initial);
  glider.Start();
  EXPECT_EQ(glider.Generation(), 0);
  glider.Update(0.1f);
  EXPECT_EQ(glider.ProjectedCells(), initial);
}
