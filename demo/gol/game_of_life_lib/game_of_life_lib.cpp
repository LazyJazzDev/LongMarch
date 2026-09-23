#include "game_of_life_lib.h"

#include <vector>

void update_step(int width, int height, uint8_t *buffer) {
  // buffer[y * width + x] is the state of cell (x, y); cells outside the grid are dead.
  std::vector<uint8_t> previous(buffer, buffer + width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int alive_neighbors = 0;
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          const int nx = x + dx;
          const int ny = y + dy;
          if ((dx || dy) && 0 <= nx && nx < width && 0 <= ny && ny < height && previous[ny * width + nx]) {
            alive_neighbors++;
          }
        }
      }
      auto &cell = buffer[y * width + x];
      if (alive_neighbors == 3) {
        cell = 1;
      } else if (alive_neighbors != 2) {
        cell = 0;
      }
    }
  }
}
