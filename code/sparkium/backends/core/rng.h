#pragma once
// Portable mirror of code/sparkium/shaders/random.hlsli.  The Sobol table is
// produced by grassland::SobolTableGen, exactly like Core::LoadPublicBuffers.
//
// The shader reads `sobol_table.Load((samp * 1024 + dim) * 4)`, i.e. the uint
// at index samp * 1024 + dim; the recorded vector is passed to the offline
// backends unchanged.

#include "sparkium/backends/core/structs.h"

namespace sparkium::backends {

SPARKIUM_HD inline uint32_t WangHash(uint32_t &seed) {
  seed = uint32_t(seed ^ uint32_t(61)) ^ uint32_t(seed >> uint32_t(16));
  seed *= uint32_t(9);
  seed = seed ^ (seed >> 4);
  seed *= uint32_t(0x27d4eb2d);
  seed = seed ^ (seed >> 15);
  return seed;
}

SPARKIUM_HD inline uint32_t WangHashS(uint32_t seed) {
  seed = uint32_t(seed ^ uint32_t(61)) ^ uint32_t(seed >> uint32_t(16));
  seed *= uint32_t(9);
  seed = seed ^ (seed >> 4);
  seed *= uint32_t(0x27d4eb2d);
  seed = seed ^ (seed >> 15);
  return seed;
}

SPARKIUM_HD inline RandomDevice InitRandomSeed(uint32_t x, uint32_t y, uint32_t s, const uint32_t *sobol_table) {
  RandomDevice random_device;
  random_device.offset = WangHashS(WangHashS(x) ^ y);
  random_device.seed = WangHashS(random_device.offset ^ s);
  random_device.dim = 0;
  random_device.samp = s;
  (void)sobol_table;
  return random_device;
}

SPARKIUM_HD inline uint32_t SobolUint(RandomDevice &random_device, const uint32_t *sobol_table) {
  uint32_t result = sobol_table[random_device.samp * 1024 + random_device.dim] ^ WangHash(random_device.offset);
  random_device.dim++;
  return result;
}

SPARKIUM_HD inline uint32_t RandomUint(RandomDevice &random_device, const uint32_t *sobol_table) {
  if (random_device.dim < 1024 && random_device.samp < 65536)
    return SobolUint(random_device, sobol_table);
  return WangHash(random_device.seed);
}

SPARKIUM_HD inline float RandomFloat(RandomDevice &rd, const uint32_t *sobol_table) {
  return float(RandomUint(rd, sobol_table)) / 4294967296.0f;
}

// `float2(RandomFloat(rd), RandomFloat(rd))` as it appears in the shaders. The
// two calls in a C++ argument list are unsequenced, so a compiler is free to
// advance the Sobol dimension in the opposite order and hand the CPU backend a
// different sample than the HLSL reference and the CUDA backend draw. Reading
// the pair through statements fixes the order for every compiler.
SPARKIUM_HD inline float2 RandomFloat2(RandomDevice &rd, const uint32_t *sobol_table) {
  float x = RandomFloat(rd, sobol_table);
  float y = RandomFloat(rd, sobol_table);
  return float2(x, y);
}

SPARKIUM_HD inline float2 RandomOnCircle(RandomDevice &rd, const uint32_t *sobol_table) {
  float theta = RandomFloat(rd, sobol_table) * SPARKIUM_PI * 2.0f;
  return float2(sinf(theta), cosf(theta));
}

SPARKIUM_HD inline float2 RandomInCircle(RandomDevice &rd, const uint32_t *sobol_table) {
  float2 circle = RandomOnCircle(rd, sobol_table);
  float radius = sqrtf(RandomFloat(rd, sobol_table));
  return circle * radius;
}

SPARKIUM_HD inline float3 RandomOnSphere(RandomDevice &rd, const uint32_t *sobol_table) {
  float z = RandomFloat(rd, sobol_table) * 2.0f - 1.0f;
  float xy = sqrtf(1.0f - z * z);
  float2 circle = RandomOnCircle(rd, sobol_table);
  return float3(xy * circle.x, xy * circle.y, z);
}

SPARKIUM_HD inline float3 RandomInSphere(RandomDevice &rd, const uint32_t *sobol_table) {
  float3 sphere = RandomOnSphere(rd, sobol_table);
  float radius = powf(RandomFloat(rd, sobol_table), 0.3333333333333333333f);
  return sphere * radius;
}

SPARKIUM_HD inline void sample_cos_hemisphere(const float3 &N,
                                              float r1,
                                              float r2,
                                              float3 &omega_in,
                                              float &pdf);

SPARKIUM_HD inline void SampleCosHemisphere(RandomDevice &rd,
                                            const float3 &N,
                                            float3 &omega_in,
                                            float &pdf,
                                            const uint32_t *sobol_table) {
  float r1 = RandomFloat(rd, sobol_table);
  float r2 = RandomFloat(rd, sobol_table);
  sample_cos_hemisphere(N, r1, r2, omega_in, pdf);
}

}  // namespace sparkium::backends
