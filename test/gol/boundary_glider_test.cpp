#include "../../demo/gol/boundary_glider.h"

#include <gtest/gtest.h>

#include "../../demo/gol/game_of_life_lib/game_of_life_lib.h"

TEST(BoundaryGlider, RecordedFramesMatchEmptySpaceEvolutionAndMoveUpRight) {
  BoundaryGlider glider;
  std::array<uint8_t, 16> initial{};
  for (int index : {4, 5, 6, 10, 13})
    initial[index] = 1;
  EXPECT_EQ(glider.ProjectedCells(), initial);
  constexpr int size = 24, origin = 8;
  std::array<uint8_t, size * size> space{};
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x)
      space[(origin + y) * size + origin + x] = initial[y * 4 + x];
  glider.Start();
  glider.Update(0.29f);
  EXPECT_EQ(glider.Generation(), 0);
  for (int generation = 1; generation <= 16; ++generation) {
    glider.Update(generation == 1 ? 0.02f : 0.13f);
    update_step(size, size, space.data(), BoundaryMode::kFixed);
    std::array<uint8_t, 16> expected{};
    for (int y = 0; y < size; ++y)
      for (int x = 0; x < size; ++x)
        if (space[y * size + x])
          expected[(y % 4) * 4 + x % 4] = 1;
    EXPECT_EQ(glider.ProjectedCells(), expected) << generation;
    EXPECT_EQ(std::count(expected.begin(), expected.end(), 1), 5);
    EXPECT_EQ(glider.Playing(), generation < 16);
    if (generation % 4 == 0) {
      std::array<uint8_t, 16> translated{};
      const int shift = generation / 4;
      for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
          translated[((y - shift + 4) % 4) * 4 + (x + shift) % 4] = initial[y * 4 + x];
      EXPECT_EQ(expected, translated);
    }
  }
  glider.Update(100.0f);
  EXPECT_EQ(glider.Generation(), 16);
  EXPECT_EQ(glider.ProjectedCells(), initial);
}

TEST(BoundaryGlider, ResetAndRestartDoNotRetainAnimationState) {
  BoundaryGlider glider;
  const auto initial = glider.ProjectedCells();
  glider.Start();
  glider.Update(5.0f);
  EXPECT_EQ(glider.Generation(), 1);
  glider.Reset();
  EXPECT_FALSE(glider.Playing());
  EXPECT_EQ(glider.ProjectedCells(), initial);
  glider.Start();
  EXPECT_EQ(glider.Generation(), 0);
  glider.Update(0.1f);
  EXPECT_EQ(glider.ProjectedCells(), initial);
}
