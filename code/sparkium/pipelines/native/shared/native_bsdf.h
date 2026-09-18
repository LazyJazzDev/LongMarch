#pragma once

// Port of `bsdf/principled_util.hlsli`, `bsdf/lambertian.hlsli` and
// `bsdf/specular.hlsli`. HLSL `out`/`inout` parameters become C++ references,
// so the uninitialised-on-early-return behaviour of the originals is
// preserved exactly.

#include "native_geometry.h"

namespace sparkium::native {

LM_DEVICE_FUNC inline float fresnel_dielectric(float eta,
                                               const float3 &N,
                                               const float3 &I,
                                               float3 &R,
                                               float3 &T,
                                               bool &is_inside) {
  float cos = glm::dot(N, I), neta;
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

  const float arg = 1 - (neta * neta * (1 - (cos * cos)));
  if (arg < 0) {
    T = make_float3(0.0f, 0.0f, 0.0f);
    return 1;  // total internal reflection
  } else {
    const float dnp = ::fmaxf(::sqrtf(arg), 1e-7f);
    const float nK = (neta * cos) - dnp;
    T = -(neta * I) + (nK * Nn);
    // compute Fresnel terms
    const float cosTheta1 = cos;  // N.R
    const float cosTheta2 = -glm::dot(Nn, T);
    const float pPara = (cosTheta1 - eta * cosTheta2) / (cosTheta1 + eta * cosTheta2);
    const float pPerp = (eta * cosTheta1 - cosTheta2) / (eta * cosTheta1 + cosTheta2);
    return 0.5f * (pPara * pPara + pPerp * pPerp);
  }
}

LM_DEVICE_FUNC inline float fresnel_dielectric_cos(float cosi, float eta) {
  const float c = ::fabsf(cosi);
  float g = eta * eta - 1 + c * c;
  if (g > 0) {
    g = ::sqrtf(g);
    const float A = (g - c) / (g + c);
    const float B = (c * (g + c) - 1) / (c * (g - c) + 1);
    return 0.5f * A * A * (1 + B * B);
  }
  return 1.0f;
}

LM_DEVICE_FUNC inline float schlick_fresnel(float u) {
  const float m = saturatef(1.0f - u);
  const float m2 = m * m;
  return m2 * m2 * m;  // pow(m, 5)
}

LM_DEVICE_FUNC inline Spectrum interpolate_fresnel_color(const float3 &L,
                                                         const float3 &H,
                                                         float ior,
                                                         float F0,
                                                         const Spectrum &cspec0) {
  /* Calculate the fresnel interpolation factor
   * The value from fresnel_dielectric_cos(...) has to be normalized because
   * the cspec0 keeps the F0 color
   */
  const float F0_norm = 1.0f / (1.0f - F0);
  const float FH = (fresnel_dielectric_cos(glm::dot(L, H), ior) - F0) * F0_norm;

  /* Blend between white and a specular color with respect to the fresnel */
  return cspec0 * (1.0f - FH) + make_float3(FH);
}

LM_DEVICE_FUNC inline float D_GTR1(float NdotH, float alpha) {
  if (alpha >= 1.0f)
    return INV_PI;
  const float alpha2 = alpha * alpha;
  const float t = 1.0f + (alpha2 - 1.0f) * NdotH * NdotH;
  return (alpha2 - 1.0f) / (PI * ::logf(alpha2) * t);
}

LM_DEVICE_FUNC inline float fast_ierff(float x) {
  /* From: Approximating the `erfinv` function by Mike Giles. */
  /* To avoid trouble at the limit, clamp input to 1-epsilon. */
  float a = ::fabsf(x);
  if (a > 0.99999994f) {
    a = 0.99999994f;
  }
  float w = -::logf((1.0f - a) * (1.0f + a)), p;
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
    w = ::sqrtf(w) - 3.0f;
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

LM_DEVICE_FUNC inline float fast_erff(float x) {
  /* Examined 1082130433 values of erff on [0,4]: 1.93715e-06 max error. */
  /* Abramowitz and Stegun, 7.1.28. */
  const float a1 = 0.0705230784f;
  const float a2 = 0.0422820123f;
  const float a3 = 0.0092705272f;
  const float a4 = 0.0001520143f;
  const float a5 = 0.0002765672f;
  const float a6 = 0.0000430638f;
  const float a = ::fabsf(x);
  if (a >= 12.3f) {
    return copysignf_hlsl(1.0f, x);
  }
  const float b = 1.0f - (1.0f - a); /* Crush denormals. */
  const float r = madd(madd(madd(madd(madd(madd(a6, b, a5), b, a4), b, a3), b, a2), b, a1), b, 1.0f);
  const float s = r * r; /* ^2 */
  const float t = s * s; /* ^4 */
  const float u = t * t; /* ^8 */
  const float v = u * u; /* ^16 */
  return copysignf_hlsl(1.0f - 1.0f / v, x);
}

// ---------------------------------------------------------------------------
// bsdf/lambertian.hlsli
// ---------------------------------------------------------------------------
LM_DEVICE_FUNC inline float3 EvalLambertianBSDF(const float3 &base_color,
                                                const float3 &N,
                                                const float3 &L,
                                                float &pdf) {
  const float cos_pi = ::fmaxf(glm::dot(N, L), 0.0f) * INV_PI;
  pdf = cos_pi;
  return cos_pi * base_color;
}

LM_DEVICE_FUNC inline void SampleLambertianBSDF(const SceneView &scene,
                                                const float3 &base_color,
                                                RandomDevice &rd,
                                                const HitRecord &hit_record,
                                                float3 &eval,
                                                float3 &L,
                                                float &pdf) {
  SampleCosHemisphere(scene, rd, hit_record.normal, L, pdf);
  if (glm::dot(hit_record.geom_normal, L) > 0.0f) {
    eval = pdf * base_color;
  } else {
    eval = float3{0, 0, 0};
  }
}

// ---------------------------------------------------------------------------
// bsdf/specular.hlsli
// ---------------------------------------------------------------------------
LM_DEVICE_FUNC inline void SampleSpecularBSDF(const float3 &base_color,
                                              const float3 &direction,
                                              const float3 &normal,
                                              const float3 &geom_normal,
                                              float3 &eval,
                                              float3 &L,
                                              float &pdf) {
  L = reflect(direction, normal);
  pdf = 1e6f;
  if (glm::dot(geom_normal, L) > 0.0f) {
    eval = base_color;
  } else {
    eval = float3{0, 0, 0};
  }
}

}  // namespace sparkium::native
