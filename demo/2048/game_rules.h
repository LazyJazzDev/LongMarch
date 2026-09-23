#pragma once

#include <array>
#include <optional>
#include <random>
#include <utility>

// Board geometry shared by the game and the autoplay strategy: x counts columns
// from the left and y rows from the bottom, matching how update_step moves
// blocks, and a cell index packs both into one number.
inline constexpr int kBoardSize = 4;
inline constexpr int kBoardCells = kBoardSize * kBoardSize;

inline constexpr int BoardCell(int x, int y) {
  return y * kBoardSize + x;
}

using CellFlags = std::array<bool, kBoardCells>;

// The spawn rule of the game: one empty cell is picked uniformly, then filled
// with a 2 with probability 0.9 and with a 4 otherwise. The interactive game
// and the offline benchmark call the same function with their own generator, so
// autoplay never touches the randomness of the game.
std::optional<std::pair<int, int>> PickSpawnCell(std::mt19937 &random, const CellFlags &occupied, int *number);
