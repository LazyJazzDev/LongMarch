#pragma once
// Portable mirror of the sampling helpers in
// code/sparkium/shaders/common.hlsli.

#include "sparkium/backends/core/hlsl_math.h"
#include "sparkium/backends/core/structs.h"

namespace sparkium::backends {

SPARKIUM_HD inline void MakeOrthonormals(const float3 &N, float3 &a, float3 &b) {
  if (N.x != N.y || N.x != N.z)
    a = float3(N.z - N.y, N.x - N.z, N.y - N.x);
  else
    a = float3(N.z - N.y, N.x + N.z, -N.y - N.x);

  a = device::normalize(a);
  b = device::cross(N, a);
}

SPARKIUM_HD inline void sample_cos_hemisphere(const float3 &N,
                                              float r1,
                                              float r2,
                                              float3 &omega_in,
                                              float &pdf) {
  r1 *= SPARKIUM_PI * 2.0f;
  float3 T, B;
  MakeOrthonormals(N, T, B);
  omega_in = float3(float2(sinf(r1), cosf(r1)) * sqrtf(1.0f - r2), sqrtf(r2));
  pdf = omega_in.z * SPARKIUM_INV_PI;
  omega_in = device::mul(omega_in, float3x3(T, B, N));
}

SPARKIUM_HD inline void make_orthonormals_tangent(const float3 &N,
                                                  const float3 &T,
                                                  float3 &a,
                                                  float3 &b) {
  b = device::normalize(device::cross(N, T));
  a = device::cross(b, N);
}

}  // namespace sparkium::backends
