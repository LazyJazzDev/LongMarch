#pragma once
// Portable port of code/sparkium/shaders/bsdf/principled_util.hlsli.
//
// Cycles-derived helpers shared by every principled closure. Kept close to the
// HLSL so both files stay comparable.

#include "sparkium/backends/core/sampling.h"
#include "sparkium/backends/core/structs.h"

namespace sparkium::backends {

using device::make_float3;

#define SPARKIUM_LABEL_NONE 0
#define SPARKIUM_LABEL_TRANSMIT 1
#define SPARKIUM_LABEL_REFLECT 2
#define SPARKIUM_LABEL_DIFFUSE 4
#define SPARKIUM_LABEL_GLOSSY 8
#define SPARKIUM_LABEL_SINGULAR 16
#define SPARKIUM_LABEL_TRANSPARENT 32
#define SPARKIUM_LABEL_VOLUME_SCATTER 64
#define SPARKIUM_LABEL_TRANSMIT_TRANSPARENT 128
#define SPARKIUM_LABEL_SUBSURFACE_SCATTER 256

#define SPARKIUM_CLOSURE_WEIGHT_CUTOFF 1e-5f

// PREPARE_BSDF from principled_util.hlsli.
#define SPARKIUM_PREPARE_BSDF(sc, weight_in)                 \
  do {                                                       \
    float3 weight_spectrum = (weight_in);                    \
    weight_spectrum = device::max(weight_spectrum, float3(0.0f)); \
    const float sample_weight = fabsf(average(weight_spectrum)); \
    if (sample_weight >= SPARKIUM_CLOSURE_WEIGHT_CUTOFF) {   \
      (sc).weight = weight_spectrum;                         \
      (sc).sample_weight = sample_weight;                    \
    } else {                                                 \
      (sc).sample_weight = 0.0f;                             \
    }                                                        \
  } while (false)

SPARKIUM_HD inline float average(const float3 &v) {
  return (v.x + v.y + v.z) / 3.0f;
}

SPARKIUM_HD inline float fresnel_dielectric(float eta,
                                            const float3 &N,
                                            const float3 &I,
                                            float3 &R,
                                            float3 &T,
                                            bool &is_inside) {
  float cos = device::dot(N, I), neta;
  float3 Nn;

  // check which side of the surface we are on
  if (cos > 0) {
    // we are on the outside of the surface, going in
    neta = 1 / eta;
    Nn = N;
    is_inside = false;
  } else {
    // we are inside the surface
    cos = -cos;
    neta = eta;
    Nn = -N;
    is_inside = true;
  }

  // compute reflection
  R = (2 * cos) * Nn - I;

  float arg = 1 - (neta * neta * (1 - (cos * cos)));
  if (arg < 0) {
    T = make_float3(0.0f, 0.0f, 0.0f);
    return 1;  // total internal reflection
  } else {
    float dnp = device::max(sqrtf(arg), 1e-7f);
    float nK = (neta * cos) - dnp;
    T = -(neta * I) + (nK * Nn);
    // compute Fresnel terms
    float cosTheta1 = cos;  // N.R
    float cosTheta2 = -device::dot(Nn, T);
    float pPara = (cosTheta1 - eta * cosTheta2) / (cosTheta1 + eta * cosTheta2);
    float pPerp = (eta * cosTheta1 - cosTheta2) / (eta * cosTheta1 + cosTheta2);
    return 0.5f * (pPara * pPara + pPerp * pPerp);
  }
}

SPARKIUM_HD inline float fresnel_dielectric_cos(float cosi, float eta) {
  float c = fabsf(cosi);
  float g = eta * eta - 1 + c * c;
  if (g > 0) {
    g = sqrtf(g);
    float A = (g - c) / (g + c);
    float B = (c * (g + c) - 1) / (c * (g - c) + 1);
    return 0.5f * A * A * (1 + B * B);
  }
  return 1.0f;
}

SPARKIUM_HD inline float schlick_fresnel(float u) {
  float m = device::clamp(1.0f - u, 0.0f, 1.0f);
  float m2 = m * m;
  return m2 * m2 * m;  // pow(m, 5)
}

SPARKIUM_HD inline float3 interpolate_fresnel_color(const float3 &L,
                                                    const float3 &H,
                                                    float ior,
                                                    float F0,
                                                    const float3 &cspec0) {
  /* Calculate the fresnel interpolation factor
   * The value from fresnel_dielectric_cos(...) has to be normalized because
   * the cspec0 keeps the F0 color
   */
  float F0_norm = 1.0f / (1.0f - F0);
  float FH = (fresnel_dielectric_cos(device::dot(L, H), ior) - F0) * F0_norm;

  /* Blend between white and a specular color with respect to the fresnel */
  return cspec0 * (1.0f - FH) + make_float3(FH);
}

SPARKIUM_HD inline float safe_sqrtf(float f) {
  return sqrtf(device::max(f, 0.0f));
}

SPARKIUM_HD inline float3 rotate_around_axis(const float3 &p, const float3 &axis, float angle) {
  float costheta = cosf(angle);
  float sintheta = sinf(angle);
  float3 r;

  r.x = ((costheta + (1 - costheta) * axis.x * axis.x) * p.x) +
        (((1 - costheta) * axis.x * axis.y - axis.z * sintheta) * p.y) +
        (((1 - costheta) * axis.x * axis.z + axis.y * sintheta) * p.z);

  r.y = (((1 - costheta) * axis.x * axis.y + axis.z * sintheta) * p.x) +
        ((costheta + (1 - costheta) * axis.y * axis.y) * p.y) +
        (((1 - costheta) * axis.y * axis.z - axis.x * sintheta) * p.z);

  r.z = (((1 - costheta) * axis.x * axis.z - axis.y * sintheta) * p.x) +
        (((1 - costheta) * axis.y * axis.z + axis.x * sintheta) * p.y) +
        ((costheta + (1 - costheta) * axis.z * axis.z) * p.z);

  return r;
}

SPARKIUM_HD inline float D_GTR1(float NdotH, float alpha) {
  if (alpha >= 1.0f)
    return SPARKIUM_INV_PI;
  float alpha2 = alpha * alpha;
  float t = 1.0f + (alpha2 - 1.0f) * NdotH * NdotH;
  return (alpha2 - 1.0f) / (SPARKIUM_PI * logf(alpha2) * t);
}

SPARKIUM_HD inline float madd(const float a, const float b, const float c) {
  return a * b + c;
}

SPARKIUM_HD inline float fast_ierff(float x) {
  /* From: Approximating the `erfinv` function by Mike Giles. */
  /* To avoid trouble at the limit, clamp input to 1-epsilon. */
  float a = fabsf(x);
  if (a > 0.99999994f) {
    a = 0.99999994f;
  }
  float w = -logf((1.0f - a) * (1.0f + a)), p;
  if (w < 5.0f) {
    w = w - 2.5f;
    p = 2.81022636e-08f;
    p = madd(p, w, 3.43273939e-07f);
    p = madd(p, w, -3.5233877e-06f);
    p = madd(p, w, -4.39150654e-06f);
    p = madd(p, w, 0.00021858087f);
    p = madd(p, w, -0.00125372503f);
    p = madd(p, w, -0.00417768164f);
    p = madd(p, w, 0.246640727f);
    p = madd(p, w, 1.50140941f);
  } else {
    w = sqrtf(w) - 3.0f;
    p = -0.000200214257f;
    p = madd(p, w, 0.000100950558f);
    p = madd(p, w, 0.00134934322f);
    p = madd(p, w, -0.00367342844f);
    p = madd(p, w, 0.00573950773f);
    p = madd(p, w, -0.0076224613f);
    p = madd(p, w, 0.00943887047f);
    p = madd(p, w, 1.00167406f);
    p = madd(p, w, 2.83297682f);
  }
  return p * x;
}

SPARKIUM_HD inline float copysignf_(float x, float y) {
  if (x * y < 0.0) {
    x = -x;
  }
  return x;
}

SPARKIUM_HD inline float fast_erff(float x) {
  /* Examined 1082130433 values of erff on [0,4]: 1.93715e-06 max error. */
  /* Abramowitz and Stegun, 7.1.28. */
  const float a1 = 0.0705230784f;
  const float a2 = 0.0422820123f;
  const float a3 = 0.0092705272f;
  const float a4 = 0.0001520143f;
  const float a5 = 0.0002765672f;
  const float a6 = 0.0000430638f;
  const float a = fabsf(x);
  if (a >= 12.3f) {
    return copysignf_(1.0f, x);
  }
  const float b = 1.0f - (1.0f - a); /* Crush denormals. */
  const float r = madd(madd(madd(madd(madd(madd(a6, b, a5), b, a4), b, a3), b, a2), b, a1), b, 1.0f);
  const float s = r * r; /* ^2 */
  const float t = s * s; /* ^4 */
  const float u = t * t; /* ^8 */
  const float v = u * u; /* ^16 */
  return copysignf_(1.0f - 1.0f / v, x);
}

}  // namespace sparkium::backends
