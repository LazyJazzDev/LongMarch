#pragma once

#include <cstdint>

// Advance a 0/1 grid in place with periodic boundaries on both axes.
// All eight directional offsets count, including repeated cells on size-2 axes.
void update_step(int width, int height, uint8_t *buffer);
