#include "game_of_life_lib.h"

#include <vector>

void update_step(int width, int height, uint8_t *buffer, BoundaryMode mode) {
  // buffer[y * width + x] is the state of cell (x, y).
  // Read the current generation from bit 0 and stage the next one in bit 1.
  // Each cell sums the vertical triples of its column and both neighbor columns,
  // so the 3x3 window needs no per-offset boundary branches. Periodic neighbors
  // wrap, repeating the other row or column on size-2 axes; fixed neighbors
  // outside the grid count as dead.
  const bool periodic = mode == BoundaryMode::kPeriodic;
  std::vector<uint8_t> column(width + 2);
  for (int y = 0; y < height; y++) {
    const uint8_t *row = buffer + y * width;
    const uint8_t *above = y > 0 ? row - width : periodic ? buffer + (height - 1) * width : nullptr;
    const uint8_t *below = y + 1 < height ? row + width : periodic ? buffer : nullptr;
    // column[x + 1] holds the vertical triple sum of column x.
    for (int x = 0; x < width; x++)
      column[x + 1] = (row[x] & 1) + (above ? above[x] & 1 : 0) + (below ? below[x] & 1 : 0);
    column[0] = periodic ? column[width] : 0;
    column[width + 1] = periodic ? column[1] : 0;
    uint8_t *cells = buffer + y * width;
    for (int x = 0; x < width; x++) {
      const uint8_t alive = cells[x] & 1;
      const int alive_neighbors = column[x] + column[x + 1] + column[x + 2] - alive;
      const uint8_t next = alive_neighbors == 3 || (alive && alive_neighbors == 2);
      cells[x] = alive | (next << 1);
    }
  }
  // Commit together, restoring the public 0/1 representation.
  for (int i = 0; i < width * height; ++i)
    buffer[i] >>= 1;
}
