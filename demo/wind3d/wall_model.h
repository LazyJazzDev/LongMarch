#pragma once
#include <cmath>
#ifdef __CUDACC__
#define WIND3D_WALL_HD __host__ __device__ inline
#else
#define WIND3D_WALL_HD inline
#endif

namespace wind3d {
// Spalding's single law of the wall, y+ as a function of u+.
WIND3D_WALL_HD float SpaldingYPlus(float up, float &derivative) {
  constexpr float kappa = 0.41f, offset = 0.1185f;  // exp(-kappa * 5.2)
  float k = kappa * up, e = expf(k);
  derivative = 1 + offset * kappa * (e - 1 - k - 0.5f * k * k);
  return up + offset * (e - 1 - k - 0.5f * k * k - k * k * k / 6);
}

// Shared log-law evaluation for uniform and adaptive load integration.
WIND3D_WALL_HD float WallEffectiveViscosity(float molecular_viscosity, float speed, float distance) {
  float reynolds = speed * distance / molecular_viscosity;
  if (!(reynolds > 1))
    return molecular_viscosity;
  // Solve u+ * y+(u+) = U y / nu, which increases monotonically with u+.
  float up = fminf(sqrtf(reynolds), 2.44f * logf(reynolds) + 5);
  for (int i = 0; i < 24; ++i) {
    float derivative, yp = SpaldingYPlus(up, derivative);
    float next = up - (up * yp - reynolds) / (yp + up * derivative);
    up = fminf(fmaxf(next, 0.5f * up), 2 * up);
  }
  return molecular_viscosity * reynolds / (up * up);
}

}  // namespace wind3d

#undef WIND3D_WALL_HD
