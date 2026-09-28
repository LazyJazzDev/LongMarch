#pragma once

#include <cstdint>

enum class BoundaryMode { kFixed, kPeriodic };

// Advance a 0/1 grid in place. Fixed mode treats outside cells as dead.
// Periodic mode counts all eight offsets, including repeated cells on size-2 axes.
void update_step(int width, int height, uint8_t *buffer, BoundaryMode mode = BoundaryMode::kPeriodic);
