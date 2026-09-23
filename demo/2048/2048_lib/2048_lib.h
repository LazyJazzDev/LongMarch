#pragma once

#include <cstdint>

struct Block {
  int x;
  int y;
  int number;
};

enum class Direction { kUp, kDown, kLeft, kRight };

void update_step(int pos0, Block *pos1, Direction direction);
