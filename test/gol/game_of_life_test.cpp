#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "../../demo/gol/game_of_life_lib/game_of_life_lib.h"

namespace {
std::vector<uint8_t> ReferenceStep(int width, int height, const std::vector<uint8_t> &cells) {
  std::vector<uint8_t> next(cells.size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int neighbors = 0;
      for (int ny = std::max(0, y - 1); ny <= std::min(height - 1, y + 1); ++ny)
        for (int nx = std::max(0, x - 1); nx <= std::min(width - 1, x + 1); ++nx)
          if (nx != x || ny != y)
            neighbors += cells[ny * width + nx];
      const int index = y * width + x;
      next[index] = neighbors == 3 || (cells[index] && neighbors == 2);
    }
  }
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
