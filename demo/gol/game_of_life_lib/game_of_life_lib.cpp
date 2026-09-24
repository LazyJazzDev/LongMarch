#include "game_of_life_lib.h"

void update_step(int width, int height, uint8_t *buffer) {
  // buffer[y * width + x] is the state of cell (x, y); both axes wrap periodically.
  // Read the current generation from bit 0 and stage the next one in bit 1.
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int alive_neighbors = 0;
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          const int nx = x + dx < 0 ? width - 1 : (x + dx == width ? 0 : x + dx);
          const int ny = y + dy < 0 ? height - 1 : (y + dy == height ? 0 : y + dy);
          if (dx || dy) {
            alive_neighbors += buffer[ny * width + nx] & 1;
          }
        }
      }
      auto &cell = buffer[y * width + x];
      const uint8_t alive = cell & 1;
      const uint8_t next = alive_neighbors == 3 || (alive && alive_neighbors == 2);
      cell = alive | (next << 1);
    }
  }
  // Commit together, restoring the public 0/1 representation.
  for (int i = 0; i < width * height; ++i)
    buffer[i] >>= 1;
}
