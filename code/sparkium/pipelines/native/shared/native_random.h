#pragma once

// Port of `shaders/random.hlsli`. The Sobol table is the very same
// `SobolTableGen(65536, 1024, ...)` blob the graphics core uploads, so a given
// (pixel, sample, dimension) triple yields bit-identical numbers on every
// backend.

#include "native_scene.h"

namespace sparkium::native {

LM_DEVICE_FUNC inline uint32_t WangHash(uint32_t &seed) {
  seed = uint32_t(seed ^ uint32_t(61)) ^ uint32_t(seed >> uint32_t(16));
  seed *= uint32_t(9);
  seed = seed ^ (seed >> 4);
  seed *= uint32_t(0x27d4eb2d);
  seed = seed ^ (seed >> 15);
  return seed;
}

LM_DEVICE_FUNC inline uint32_t WangHashS(uint32_t seed) {
  seed = uint32_t(seed ^ uint32_t(61)) ^ uint32_t(seed >> uint32_t(16));
  seed *= uint32_t(9);
  seed = seed ^ (seed >> 4);
  seed *= uint32_t(0x27d4eb2d);
  seed = seed ^ (seed >> 15);
  return seed;
}

LM_DEVICE_FUNC inline RandomDevice InitRandomSeed(uint32_t x, uint32_t y, uint32_t s) {
  RandomDevice random_device;
  random_device.offset = WangHashS(WangHashS(x) ^ y);
  random_device.seed = WangHashS(random_device.offset ^ s);
  random_device.dim = 0;
  random_device.samp = s;
  return random_device;
}

LM_DEVICE_FUNC inline uint32_t SobolUint(const SceneView &scene, RandomDevice &random_device) {
  const uint32_t result =
      scene.sobol_table.Load((random_device.samp * 1024 + random_device.dim) * 4) ^ WangHash(random_device.offset);
  random_device.dim++;
  return result;
}

LM_DEVICE_FUNC inline uint32_t RandomUint(const SceneView &scene, RandomDevice &random_device) {
  if (random_device.dim < 1024 && random_device.samp < 65536)
    return SobolUint(scene, random_device);
  return WangHash(random_device.seed);
}

LM_DEVICE_FUNC inline float RandomFloat(const SceneView &scene, RandomDevice &rd) {
  return float(RandomUint(scene, rd)) / 4294967296.0f;
}

LM_DEVICE_FUNC inline float2 RandomOnCircle(const SceneView &scene, RandomDevice &rd) {
  const float theta = RandomFloat(scene, rd) * PI * 2.0f;
  return float2{::sinf(theta), ::cosf(theta)};
}

LM_DEVICE_FUNC inline float2 RandomInCircle(const SceneView &scene, RandomDevice &rd) {
  return RandomOnCircle(scene, rd) * ::sqrtf(RandomFloat(scene, rd));
}

LM_DEVICE_FUNC inline float3 RandomOnSphere(const SceneView &scene, RandomDevice &rd) {
  const float z = RandomFloat(scene, rd) * 2.0f - 1.0f;
  const float xy = ::sqrtf(1.0f - z * z);
  const float2 circle = RandomOnCircle(scene, rd) * xy;
  return float3{circle.x, circle.y, z};
}

LM_DEVICE_FUNC inline float3 RandomInSphere(const SceneView &scene, RandomDevice &rd) {
  return RandomOnSphere(scene, rd) * ::powf(RandomFloat(scene, rd), 0.3333333333333333333f);
}

LM_DEVICE_FUNC inline void SampleCosHemisphere(const SceneView &scene,
                                               RandomDevice &rd,
                                               const float3 &N,
                                               float3 &omega_in,
                                               float &pdf) {
  const float r1 = RandomFloat(scene, rd);
  const float r2 = RandomFloat(scene, rd);
  sample_cos_hemisphere(N, r1, r2, omega_in, pdf);
}

}  // namespace sparkium::native
