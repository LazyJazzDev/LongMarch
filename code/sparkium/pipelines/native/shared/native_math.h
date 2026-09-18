#pragma once

// HLSL-flavoured scalar and vector helpers shared by the CPU and CUDA
// backends. Every function is compiled both by the host compiler and by nvcc
// so the shading core below has exactly one implementation.
//
// Type mapping, matching the byte layouts written by the graphics pipelines:
//   HLSL float3x4 (3 rows, 4 columns)  <->  glm::mat4x3 (4 columns, 3 rows)
//   HLSL mul(M, v4)                    <->  M * v4
//   HLSL mul(transpose(M), v3)         <->  v3 * M

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "glm/glm.hpp"
#include "grassland/util/util.h"

namespace sparkium::native {

using float2 = glm::vec2;
using float3 = glm::vec3;
using float4 = glm::vec4;
using uint2 = glm::uvec2;
using uint3 = glm::uvec3;
using uint4 = glm::uvec4;
using int2 = glm::ivec2;
using float3x4 = glm::mat4x3;  // HLSL float3x4
using Spectrum = float3;

#define SPARKIUM_PI 3.14159265358979323f
#define SPARKIUM_INV_PI 0.31830988618379067f
#define SPARKIUM_T_MIN 0.001f
#define SPARKIUM_T_MAX 1e9f
#define SPARKIUM_EPSILON 0.0001f
#define SPARKIUM_INF 1e8f

#define PI SPARKIUM_PI
#define INV_PI SPARKIUM_INV_PI
#define T_MIN SPARKIUM_T_MIN
#define T_MAX SPARKIUM_T_MAX
#define EPSILON SPARKIUM_EPSILON
#define INF SPARKIUM_INF

#define CLOSURE_WEIGHT_CUTOFF 1e-5f

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

// ---------------------------------------------------------------------------
// Bit casts
// ---------------------------------------------------------------------------
LM_DEVICE_FUNC inline float asfloat(uint32_t v) {
  float result;
  ::memcpy(&result, &v, sizeof(float));
  return result;
}

LM_DEVICE_FUNC inline float asfloat(int32_t v) {
  return asfloat(static_cast<uint32_t>(v));
}

LM_DEVICE_FUNC inline uint32_t asuint(float v) {
  uint32_t result;
  ::memcpy(&result, &v, sizeof(uint32_t));
  return result;
}

LM_DEVICE_FUNC inline int32_t asint(float v) {
  return static_cast<int32_t>(asuint(v));
}

LM_DEVICE_FUNC inline float3 asfloat(const uint3 &v) {
  return float3{asfloat(v.x), asfloat(v.y), asfloat(v.z)};
}

LM_DEVICE_FUNC inline uint3 asuint(const float3 &v) {
  return uint3{asuint(v.x), asuint(v.y), asuint(v.z)};
}

// ---------------------------------------------------------------------------
// Scalar intrinsics
// ---------------------------------------------------------------------------
LM_DEVICE_FUNC inline float saturatef(float x) {
  return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}

LM_DEVICE_FUNC inline float saturate(float x) {
  return saturatef(x);
}

LM_DEVICE_FUNC inline float3 saturate(const float3 &v) {
  return float3{saturatef(v.x), saturatef(v.y), saturatef(v.z)};
}

LM_DEVICE_FUNC inline float4 saturate(const float4 &v) {
  return float4{saturatef(v.x), saturatef(v.y), saturatef(v.z), saturatef(v.w)};
}

LM_DEVICE_FUNC inline float lerp(float a, float b, float t) {
  return a + (b - a) * t;
}

LM_DEVICE_FUNC inline float3 lerp(const float3 &a, const float3 &b, float t) {
  return a + (b - a) * t;
}

LM_DEVICE_FUNC inline float4 lerp(const float4 &a, const float4 &b, float t) {
  return a + (b - a) * t;
}

LM_DEVICE_FUNC inline float4 lerp(const float4 &a, const float4 &b, const float4 &t) {
  return a + (b - a) * t;
}

LM_DEVICE_FUNC inline float3 lerp(const float3 &a, const float3 &b, const float3 &t) {
  return a + (b - a) * t;
}

LM_DEVICE_FUNC inline float frac(float x) {
  return x - ::floorf(x);
}

LM_DEVICE_FUNC inline float2 frac(const float2 &v) {
  return float2{frac(v.x), frac(v.y)};
}

LM_DEVICE_FUNC inline float3 frac(const float3 &v) {
  return float3{frac(v.x), frac(v.y), frac(v.z)};
}

LM_DEVICE_FUNC inline float4 frac(const float4 &v) {
  return float4{frac(v.x), frac(v.y), frac(v.z), frac(v.w)};
}

LM_DEVICE_FUNC inline float step(float edge, float x) {
  return x < edge ? 0.0f : 1.0f;
}

LM_DEVICE_FUNC inline float3 step(const float3 &edge, const float3 &x) {
  return float3{step(edge.x, x.x), step(edge.y, x.y), step(edge.z, x.z)};
}

LM_DEVICE_FUNC inline float4 step(const float4 &edge, const float4 &x) {
  return float4{step(edge.x, x.x), step(edge.y, x.y), step(edge.z, x.z), step(edge.w, x.w)};
}

LM_DEVICE_FUNC inline float safe_sqrtf(float f) {
  return ::sqrtf(f > 0.0f ? f : 0.0f);
}

LM_DEVICE_FUNC inline float madd(float a, float b, float c) {
  return a * b + c;
}

LM_DEVICE_FUNC inline float copysignf_hlsl(float x, float y) {
  if (x * y < 0.0f)
    x = -x;
  return x;
}

LM_DEVICE_FUNC inline float3 make_float3(float v) {
  return float3{v, v, v};
}

LM_DEVICE_FUNC inline float3 make_float3(float x, float y, float z) {
  return float3{x, y, z};
}

LM_DEVICE_FUNC inline float average(const float3 &v) {
  return (v.x + v.y + v.z) / 3.0f;
}

LM_DEVICE_FUNC inline float max3(const float3 &v) {
  return ::fmaxf(v.x, ::fmaxf(v.y, v.z));
}

LM_DEVICE_FUNC inline bool IsFinite(float v) {
  return v == v && v < FLT_MAX && v > -FLT_MAX;
}

LM_DEVICE_FUNC inline bool AllFinite(const float3 &v) {
  return IsFinite(v.x) && IsFinite(v.y) && IsFinite(v.z);
}

LM_DEVICE_FUNC inline bool IsNaN(float v) {
  return !(v == v);
}

// ---------------------------------------------------------------------------
// Matrix helpers
// ---------------------------------------------------------------------------

// HLSL: mul(float3x4, float4)
LM_DEVICE_FUNC inline float3 mul(const float3x4 &m, const float4 &v) {
  return m * v;
}

// HLSL: mul(transpose(float3x4), float3) == mul(float4x3, float3)
LM_DEVICE_FUNC inline float4 mul_transposed(const float3x4 &m, const float3 &v) {
  return v * m;
}

// HLSL: mul(v, float3x3(row0, row1, row2))
LM_DEVICE_FUNC inline float3 mul_rows(const float3 &v, const float3 &row0, const float3 &row1, const float3 &row2) {
  return row0 * v.x + row1 * v.y + row2 * v.z;
}

LM_DEVICE_FUNC inline float3 reflect(const float3 &incident, const float3 &normal) {
  return incident - 2.0f * glm::dot(incident, normal) * normal;
}

LM_DEVICE_FUNC inline void MakeOrthonormals(const float3 &N, float3 &a, float3 &b) {
  if (N.x != N.y || N.x != N.z)
    a = float3{N.z - N.y, N.x - N.z, N.y - N.x};
  else
    a = float3{N.z - N.y, N.x + N.z, -N.y - N.x};

  a = glm::normalize(a);
  b = glm::cross(N, a);
}

LM_DEVICE_FUNC inline void sample_cos_hemisphere(const float3 &N, float r1, float r2, float3 &omega_in, float &pdf) {
  r1 *= PI * 2.0f;
  float3 T, B;
  MakeOrthonormals(N, T, B);
  const float radius = ::sqrtf(1.0f - r2);
  omega_in = float3{::sinf(r1) * radius, ::cosf(r1) * radius, ::sqrtf(r2)};
  pdf = omega_in.z * INV_PI;
  omega_in = mul_rows(omega_in, T, B, N);
}

LM_DEVICE_FUNC inline void make_orthonormals_tangent(const float3 &N, const float3 &T, float3 &a, float3 &b) {
  b = glm::normalize(glm::cross(N, T));
  a = glm::cross(b, N);
}

LM_DEVICE_FUNC inline float3 rotate_around_axis(const float3 &p, const float3 &axis, float angle) {
  const float costheta = ::cosf(angle);
  const float sintheta = ::sinf(angle);
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

}  // namespace sparkium::native
