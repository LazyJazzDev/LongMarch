#include "../../demo/gol/boundary_glider.h"

#include <gtest/gtest.h>

TEST(BoundaryGlider, EvolvesInEmptySpaceBeforeProjectingAndStopsAfterOneTrip) {
  BoundaryGlider glider;
  const auto initial = glider.ProjectedCells();
  glider.Start();
  glider.Update(0.29f);
  EXPECT_EQ(glider.Generation(), 0);
  glider.Update(0.02f);
  // The real first generation has five cells. A 4x4 torus instead has eight.
  std::array<uint8_t, 16> first{};
  for (int index : {4, 6, 9, 10, 13})
    first[index] = 1;
  EXPECT_EQ(glider.ProjectedCells(), first);
  for (int generation = 2; generation <= 16; ++generation) {
    glider.Update(0.13f);
    const auto cells = glider.ProjectedCells();
    EXPECT_EQ(std::count(cells.begin(), cells.end(), 1), 5);
    EXPECT_EQ(glider.Generation(), generation);
    EXPECT_EQ(glider.Playing(), generation < 16);
    if (generation % 4 == 0) {
      std::array<uint8_t, 16> translated{};
      const int shift = generation / 4;
      for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
          translated[((y + shift) % 4) * 4 + (x + shift) % 4] = initial[y * 4 + x];
      EXPECT_EQ(cells, translated);
    }
  }
  EXPECT_EQ(glider.ProjectedCells(), initial);
  glider.Update(100.0f);
  EXPECT_EQ(glider.Generation(), 16);
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
