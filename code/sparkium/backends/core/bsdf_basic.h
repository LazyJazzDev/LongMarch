#pragma once
// Portable port of the lambertian and specular BSDFs:
//   code/sparkium/shaders/bsdf/lambertian.hlsli
//   code/sparkium/shaders/bsdf/specular.hlsli

#include "sparkium/backends/core/rng.h"
#include "sparkium/backends/core/structs.h"

namespace sparkium::backends {

SPARKIUM_HD inline float3 EvalLambertianBSDF(const float3 &base_color,
                                            const float3 &N,
                                            const float3 &L,
                                            float &pdf) {
  float cos_pi = device::max(device::dot(N, L), 0.0f) * SPARKIUM_INV_PI;
  pdf = cos_pi;
  return cos_pi * base_color;
}

SPARKIUM_HD inline void SampleLambertianBSDF(const float3 &base_color,
                                             RandomDevice &rd,
                                             const HitRecord &hit_record,
                                             float3 &eval,
                                             float3 &L,
                                             float &pdf,
                                             const uint32_t *sobol_table) {
  SampleCosHemisphere(rd, hit_record.normal, L, pdf, sobol_table);
  if (device::dot(hit_record.geom_normal, L) > 0.0) {
    eval = pdf * base_color;
  } else {
    eval = float3(0, 0, 0);
  }
}

SPARKIUM_HD inline void SampleSpecularBSDF(const float3 &base_color,
                                           const float3 &direction,
                                           const float3 &normal,
                                           const float3 &geom_normal,
                                           float3 &eval,
                                           float3 &L,
                                           float &pdf) {
  L = device::reflect(direction, normal);
  pdf = 1e6f;
  if (device::dot(geom_normal, L) > 0.0) {
    eval = base_color;
  } else {
    eval = float3(0, 0, 0);
  }
}

}  // namespace sparkium::backends
