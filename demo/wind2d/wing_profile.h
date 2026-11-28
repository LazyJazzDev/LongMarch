#pragma once

// Shared by CUDA, the host UI, and the embedded Slang display shader.
#ifndef WING_PROFILE_SHADER
#include <cmath>
#ifdef __CUDACC__
#define WING_PROFILE_FUNCTION __host__ __device__ inline
#else
#define WING_PROFILE_FUNCTION inline
#endif
#define WING_PROFILE_SQRT sqrtf
#else
#define WING_PROFILE_FUNCTION
#define WING_PROFILE_SQRT sqrt
#endif

#define WING_NACA0012 0
#define WING_NACA0006 1
#define WING_NACA0024 2
#define WING_CAMBERED2 3
#define WING_CAMBERED4 4
#define WING_REFLEX 5
#define WING_ELLIPSE 6
#define WING_DOUBLE_WEDGE 7
#define WING_PROFILE_COUNT 8

// Coordinates are fractions of chord. Cambered profiles use vertical thickness
// offsets, rather than normal offsets, and are deliberately not named NACA 24xx/44xx.
WING_PROFILE_FUNCTION float WingCamber(float q, int profile) {
  if (q <= 0 || q >= 1)
    return 0;
  if (profile == WING_REFLEX)
    return 0.16f * q * (1 - q) * (1 - 2 * q);
  if (profile != WING_CAMBERED2 && profile != WING_CAMBERED4)
    return 0;
  float m = profile == WING_CAMBERED2 ? 0.02f : 0.04f;
  return q < 0.4f ? m * (0.8f * q - q * q) / 0.16f : m * (0.2f + 0.8f * q - q * q) / 0.36f;
}

WING_PROFILE_FUNCTION float WingThickness(float q, int profile) {
  if (q <= 0 || q >= 1)
    return 0;
  if (profile == WING_ELLIPSE)
    return 0.12f * WING_PROFILE_SQRT(q * (1 - q));
  if (profile == WING_DOUBLE_WEDGE)
    return 0.12f * (q < 0.5f ? q : 1 - q);
  float thickness = profile == WING_NACA0006 ? 0.06f : profile == WING_NACA0024 ? 0.24f : 0.12f;
  return 5 * thickness *
         (0.2969f * WING_PROFILE_SQRT(q) - 0.1260f * q - 0.3516f * q * q + 0.2843f * q * q * q -
          0.1036f * q * q * q * q);
}

#undef WING_PROFILE_FUNCTION
#undef WING_PROFILE_SQRT
