#pragma once
#include "common.hlsli"
#ifndef Spectrum
#define Spectrum float3
#endif

float3 make_float3(float v) {
  return float3(v, v, v);
}

float3 make_float3(float x, float y, float z) {
  return float3(x, y, z);
}

#define LABEL_NONE 0
#define LABEL_TRANSMIT 1
#define LABEL_REFLECT 2
#define LABEL_DIFFUSE 4
#define LABEL_GLOSSY 8
#define LABEL_SINGULAR 16
#define LABEL_TRANSPARENT 32
#define LABEL_VOLUME_SCATTER 64
#define LABEL_TRANSMIT_TRANSPARENT 128
#define LABEL_SUBSURFACE_SCATTER 256

#define CLOSURE_WEIGHT_CUTOFF 1e-5f
#define saturatef(x) clamp(x, 0.0, 1.0)
#define saturate(x) clamp(x, 0.0, 1.0)
#define PREPARE_BSDF(sc, weight_in)                 \
  Spectrum weight_spectrum = weight_in;                      \
  weight_spectrum = max(weight_spectrum, make_float3(0.0));                  \
  const float sample_weight = abs(average(weight_spectrum)); \
  if (sample_weight >= CLOSURE_WEIGHT_CUTOFF) {     \
    sc.weight = weight_spectrum;                             \
    sc.sample_weight = sample_weight;               \
  } else {                                          \
    sc.sample_weight = 0.0;                         \
  }

float average(float3 v) {
  return (v.x + v.y + v.z) / 3.0f;
}

float fresnel_dielectric(float eta,
                         const float3 N,
                         const float3 I,
                         inout float3 R,
                         inout float3 T,
                         inout bool is_inside) {
  float cos = dot(N, I), neta;
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
    float dnp = max(sqrt(arg), 1e-7f);
    float nK = (neta * cos) - dnp;
    T = -(neta * I) + (nK * Nn);
    // compute Fresnel terms
    float cosTheta1 = cos;  // N.R
    float cosTheta2 = -dot(Nn, T);
    float pPara = (cosTheta1 - eta * cosTheta2) / (cosTheta1 + eta * cosTheta2);
    float pPerp = (eta * cosTheta1 - cosTheta2) / (eta * cosTheta1 + cosTheta2);
    return 0.5f * (pPara * pPara + pPerp * pPerp);
  }
}

float fresnel_dielectric_cos(float cosi, float eta) {
  float c = abs(cosi);
  float g = eta * eta - 1 + c * c;
  if (g > 0) {
    g = sqrt(g);
    float A = (g - c) / (g + c);
    float B = (c * (g + c) - 1) / (c * (g - c) + 1);
    return 0.5f * A * A * (1 + B * B);
  }
  return 1.0f;
}

float schlick_fresnel(float u) {
  float m = clamp(1.0f - u, 0.0f, 1.0f);
  float m2 = m * m;
  return m2 * m2 * m;  // pow(m, 5)
}

Spectrum interpolate_fresnel_color(float3 L,
                                   float3 H,
                                   float ior,
                                   float F0,
                                   Spectrum cspec0) {
  /* Calculate the fresnel interpolation factor
   * The value from fresnel_dielectric_cos(...) has to be normalized because
   * the cspec0 keeps the F0 color
   */
  float F0_norm = 1.0f / (1.0f - F0);
  float FH = (fresnel_dielectric_cos(dot(L, H), ior) - F0) * F0_norm;

  /* Blend between white and a specular color with respect to the fresnel */
  return cspec0 * (1.0f - FH) + make_float3(FH);
}

float safe_sqrtf(float f) {
  return sqrt(max(f, 0.0f));
}

float3 rotate_around_axis(float3 p, float3 axis, float angle) {
  float costheta = cos(angle);
  float sintheta = sin(angle);
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

void make_orthonormals_tangent(const float3 N,
                               const float3 T,
                               out float3 a,
                               out float3 b) {
  b = normalize(cross(N, T));
  a = cross(b, N);
}

float D_GTR1(float NdotH, float alpha) {
  if (alpha >= 1.0f)
    return INV_PI;
  float alpha2 = alpha * alpha;
  float t = 1.0f + (alpha2 - 1.0f) * NdotH * NdotH;
  return (alpha2 - 1.0f) / (PI * log(alpha2) * t);
}

float madd(const float a, const float b, const float c) {
  /* NOTE: In the future we may want to explicitly ask for a fused
   * multiply-add in a specialized version for float.
   *
   * NOTE: GCC/ICC will turn this (for float) into a FMA unless
   * explicitly asked not to, clang seems to leave the code alone.
   */
  return a * b + c;
}

float fast_ierff(float x) {
  /* From: Approximating the `erfinv` function by Mike Giles. */
  /* To avoid trouble at the limit, clamp input to 1-epsilon. */
  float a = abs(x);
  if (a > 0.99999994f) {
    a = 0.99999994f;
  }
  float w = -log((1.0f - a) * (1.0f + a)), p;
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
    w = sqrt(w) - 3.0f;
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

float copysignf(float x, float y) {
  if (x * y < 0.0) {
    x = -x;
  }
  return x;
}

float fast_erff(float x) {
  /* Examined 1082130433 values of erff on [0,4]: 1.93715e-06 max error. */
  /* Abramowitz and Stegun, 7.1.28. */
  const float a1 = 0.0705230784f;
  const float a2 = 0.0422820123f;
  const float a3 = 0.0092705272f;
  const float a4 = 0.0001520143f;
  const float a5 = 0.0002765672f;
  const float a6 = 0.0000430638f;
  const float a = abs(x);
  if (a >= 12.3f) {
    return copysignf(1.0f, x);
  }
  const float b = 1.0f - (1.0f - a); /* Crush denormals. */
  const float r =
      madd(madd(madd(madd(madd(madd(a6, b, a5), b, a4), b, a3), b, a2), b, a1),
           b, 1.0f);
  const float s = r * r; /* ^2 */
  const float t = s * s; /* ^4 */
  const float u = t * t; /* ^8 */
  const float v = u * u; /* ^16 */
  return copysignf(1.0f - 1.0f / v, x);
}
