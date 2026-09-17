#pragma once
// Minimal HLSL-compatible math used by the portable Sparkium shading core.
//
// The Sparkium shaders in code/sparkium/shaders are written in HLSL and remain
// the source of truth for the Vulkan/D3D12/Metal backends.  The CPU and CUDA
// backends compile one portable transcription of the same shading core; this
// header provides the HLSL scalar/vector semantics that transcription relies
// on (broadcasting constructors, component-wise operators, lerp/step/saturate
// and row-wise `mul`).

#include <math.h>
#include <stdint.h>

#if defined(__CUDACC__)
#define SPARKIUM_HD __host__ __device__
#else
#define SPARKIUM_HD
#endif

namespace sparkium::device {

// ---------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------

SPARKIUM_HD inline float saturatef(float v) {
  return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

SPARKIUM_HD inline float rcpf(float v) {
  return 1.0f / v;
}

SPARKIUM_HD inline float radiansf(float degrees) {
  return degrees * 0.01745329251994329577f;
}

SPARKIUM_HD inline float degreesf(float radians) {
  return radians * 57.29577951308232088f;
}

SPARKIUM_HD inline bool isnanf(float v) {
  return v != v;
}

SPARKIUM_HD inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

SPARKIUM_HD inline float asfloat(uint32_t v) {
  union {
    uint32_t u;
    float f;
  } cast;
  cast.u = v;
  return cast.f;
}

SPARKIUM_HD inline uint32_t asuint(float v) {
  union {
    float f;
    uint32_t u;
  } cast;
  cast.f = v;
  return cast.u;
}

SPARKIUM_HD inline uint32_t asuint(int v) {
  union {
    int i;
    uint32_t u;
  } cast;
  cast.i = v;
  return cast.u;
}

// ---------------------------------------------------------------------------
// Vector types
// ---------------------------------------------------------------------------

struct float2 {
  float x{0.0f};
  float y{0.0f};

  SPARKIUM_HD float2() {
  }
  SPARKIUM_HD float2(float v) : x(v), y(v) {
  }
  SPARKIUM_HD float2(float x, float y) : x(x), y(y) {
  }
  SPARKIUM_HD float &operator[](int i) {
    return i == 0 ? x : y;
  }
  SPARKIUM_HD float operator[](int i) const {
    return i == 0 ? x : y;
  }
};

struct float3 {
  float x{0.0f};
  float y{0.0f};
  float z{0.0f};

  SPARKIUM_HD float3() {
  }
  SPARKIUM_HD float3(float v) : x(v), y(v), z(v) {
  }
  SPARKIUM_HD float3(float x, float y, float z) : x(x), y(y), z(z) {
  }
  SPARKIUM_HD float3(const float2 &xy, float z) : x(xy.x), y(xy.y), z(z) {
  }
  SPARKIUM_HD float &operator[](int i) {
    return i == 0 ? x : (i == 1 ? y : z);
  }
  SPARKIUM_HD float operator[](int i) const {
    return i == 0 ? x : (i == 1 ? y : z);
  }
};

struct float4 {
  float x{0.0f};
  float y{0.0f};
  float z{0.0f};
  float w{0.0f};

  SPARKIUM_HD float4() {
  }
  SPARKIUM_HD float4(float v) : x(v), y(v), z(v), w(v) {
  }
  SPARKIUM_HD float4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {
  }
  SPARKIUM_HD float4(const float3 &xyz, float w) : x(xyz.x), y(xyz.y), z(xyz.z), w(w) {
  }
  SPARKIUM_HD float4(const float2 &xy, float z, float w) : x(xy.x), y(xy.y), z(z), w(w) {
  }
  SPARKIUM_HD float4(const float2 &xy, const float2 &zw) : x(xy.x), y(xy.y), z(zw.x), w(zw.y) {
  }
  SPARKIUM_HD float &operator[](int i) {
    return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w));
  }
  SPARKIUM_HD float operator[](int i) const {
    return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w));
  }
  SPARKIUM_HD float3 xyz() const {
    return float3(x, y, z);
  }
  SPARKIUM_HD float2 xy() const {
    return float2(x, y);
  }
};

// ---------------------------------------------------------------------------
// Vector operators
// ---------------------------------------------------------------------------

#define SPARKIUM_DEFINE_VEC2_OPERATORS(T)                                                      \
  SPARKIUM_HD inline T operator-(const T &a) {                                                 \
    return T(-a.x, -a.y);                                                                      \
  }                                                                                            \
  SPARKIUM_HD inline T operator+(const T &a, const T &b) {                                     \
    return T(a.x + b.x, a.y + b.y);                                                            \
  }                                                                                            \
  SPARKIUM_HD inline T operator-(const T &a, const T &b) {                                     \
    return T(a.x - b.x, a.y - b.y);                                                            \
  }                                                                                            \
  SPARKIUM_HD inline T operator*(const T &a, const T &b) {                                     \
    return T(a.x * b.x, a.y * b.y);                                                            \
  }                                                                                            \
  SPARKIUM_HD inline T operator/(const T &a, const T &b) {                                     \
    return T(a.x / b.x, a.y / b.y);                                                            \
  }                                                                                            \
  SPARKIUM_HD inline T operator+(const T &a, float b) {                                        \
    return T(a.x + b, a.y + b);                                                                \
  }                                                                                            \
  SPARKIUM_HD inline T operator+(float a, const T &b) {                                        \
    return T(a + b.x, a + b.y);                                                                \
  }                                                                                            \
  SPARKIUM_HD inline T operator-(const T &a, float b) {                                        \
    return T(a.x - b, a.y - b);                                                                \
  }                                                                                            \
  SPARKIUM_HD inline T operator-(float a, const T &b) {                                        \
    return T(a - b.x, a - b.y);                                                                \
  }                                                                                            \
  SPARKIUM_HD inline T operator*(const T &a, float b) {                                        \
    return T(a.x * b, a.y * b);                                                                \
  }                                                                                            \
  SPARKIUM_HD inline T operator*(float a, const T &b) {                                        \
    return T(a * b.x, a * b.y);                                                                \
  }                                                                                            \
  SPARKIUM_HD inline T operator/(const T &a, float b) {                                        \
    return T(a.x / b, a.y / b);                                                                \
  }                                                                                            \
  SPARKIUM_HD inline T operator/(float a, const T &b) {                                        \
    return T(a / b.x, a / b.y);                                                                \
  }                                                                                            \
  SPARKIUM_HD inline T &operator+=(T &a, const T &b) {                                         \
    a.x += b.x;                                                                                \
    a.y += b.y;                                                                                \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator-=(T &a, const T &b) {                                         \
    a.x -= b.x;                                                                                \
    a.y -= b.y;                                                                                \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator*=(T &a, const T &b) {                                         \
    a.x *= b.x;                                                                                \
    a.y *= b.y;                                                                                \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator/=(T &a, const T &b) {                                         \
    a.x /= b.x;                                                                                \
    a.y /= b.y;                                                                                \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator+=(T &a, float b) {                                            \
    a.x += b;                                                                                  \
    a.y += b;                                                                                  \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator-=(T &a, float b) {                                            \
    a.x -= b;                                                                                  \
    a.y -= b;                                                                                  \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator*=(T &a, float b) {                                            \
    a.x *= b;                                                                                  \
    a.y *= b;                                                                                  \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator/=(T &a, float b) {                                            \
    a.x /= b;                                                                                  \
    a.y /= b;                                                                                  \
    return a;                                                                                  \
  }

#define SPARKIUM_DEFINE_VEC3_OPERATORS(T)                                                      \
  SPARKIUM_HD inline T operator-(const T &a) {                                                 \
    return T(-a.x, -a.y, -a.z);                                                                \
  }                                                                                            \
  SPARKIUM_HD inline T operator+(const T &a, const T &b) {                                     \
    return T(a.x + b.x, a.y + b.y, a.z + b.z);                                                 \
  }                                                                                            \
  SPARKIUM_HD inline T operator-(const T &a, const T &b) {                                     \
    return T(a.x - b.x, a.y - b.y, a.z - b.z);                                                 \
  }                                                                                            \
  SPARKIUM_HD inline T operator*(const T &a, const T &b) {                                     \
    return T(a.x * b.x, a.y * b.y, a.z * b.z);                                                 \
  }                                                                                            \
  SPARKIUM_HD inline T operator/(const T &a, const T &b) {                                     \
    return T(a.x / b.x, a.y / b.y, a.z / b.z);                                                 \
  }                                                                                            \
  SPARKIUM_HD inline T operator+(const T &a, float b) {                                        \
    return T(a.x + b, a.y + b, a.z + b);                                                       \
  }                                                                                            \
  SPARKIUM_HD inline T operator+(float a, const T &b) {                                        \
    return T(a + b.x, a + b.y, a + b.z);                                                       \
  }                                                                                            \
  SPARKIUM_HD inline T operator-(const T &a, float b) {                                        \
    return T(a.x - b, a.y - b, a.z - b);                                                       \
  }                                                                                            \
  SPARKIUM_HD inline T operator-(float a, const T &b) {                                        \
    return T(a - b.x, a - b.y, a - b.z);                                                       \
  }                                                                                            \
  SPARKIUM_HD inline T operator*(const T &a, float b) {                                        \
    return T(a.x * b, a.y * b, a.z * b);                                                       \
  }                                                                                            \
  SPARKIUM_HD inline T operator*(float a, const T &b) {                                        \
    return T(a * b.x, a * b.y, a * b.z);                                                       \
  }                                                                                            \
  SPARKIUM_HD inline T operator/(const T &a, float b) {                                        \
    return T(a.x / b, a.y / b, a.z / b);                                                       \
  }                                                                                            \
  SPARKIUM_HD inline T operator/(float a, const T &b) {                                        \
    return T(a / b.x, a / b.y, a / b.z);                                                       \
  }                                                                                            \
  SPARKIUM_HD inline T &operator+=(T &a, const T &b) {                                         \
    a.x += b.x;                                                                                \
    a.y += b.y;                                                                                \
    a.z += b.z;                                                                                \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator-=(T &a, const T &b) {                                         \
    a.x -= b.x;                                                                                \
    a.y -= b.y;                                                                                \
    a.z -= b.z;                                                                                \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator*=(T &a, const T &b) {                                         \
    a.x *= b.x;                                                                                \
    a.y *= b.y;                                                                                \
    a.z *= b.z;                                                                                \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator/=(T &a, const T &b) {                                         \
    a.x /= b.x;                                                                                \
    a.y /= b.y;                                                                                \
    a.z /= b.z;                                                                                \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator+=(T &a, float b) {                                            \
    a.x += b;                                                                                  \
    a.y += b;                                                                                  \
    a.z += b;                                                                                  \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator-=(T &a, float b) {                                            \
    a.x -= b;                                                                                  \
    a.y -= b;                                                                                  \
    a.z -= b;                                                                                  \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator*=(T &a, float b) {                                            \
    a.x *= b;                                                                                  \
    a.y *= b;                                                                                  \
    a.z *= b;                                                                                  \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator/=(T &a, float b) {                                            \
    a.x /= b;                                                                                  \
    a.y /= b;                                                                                  \
    a.z /= b;                                                                                  \
    return a;                                                                                  \
  }

#define SPARKIUM_DEFINE_VEC4_OPERATORS(T)                                                      \
  SPARKIUM_HD inline T operator-(const T &a) {                                                 \
    return T(-a.x, -a.y, -a.z, -a.w);                                                          \
  }                                                                                            \
  SPARKIUM_HD inline T operator+(const T &a, const T &b) {                                     \
    return T(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);                                      \
  }                                                                                            \
  SPARKIUM_HD inline T operator-(const T &a, const T &b) {                                     \
    return T(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);                                      \
  }                                                                                            \
  SPARKIUM_HD inline T operator*(const T &a, const T &b) {                                     \
    return T(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);                                      \
  }                                                                                            \
  SPARKIUM_HD inline T operator/(const T &a, const T &b) {                                     \
    return T(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);                                      \
  }                                                                                            \
  SPARKIUM_HD inline T operator+(const T &a, float b) {                                        \
    return T(a.x + b, a.y + b, a.z + b, a.w + b);                                              \
  }                                                                                            \
  SPARKIUM_HD inline T operator+(float a, const T &b) {                                        \
    return T(a + b.x, a + b.y, a + b.z, a + b.w);                                              \
  }                                                                                            \
  SPARKIUM_HD inline T operator-(const T &a, float b) {                                        \
    return T(a.x - b, a.y - b, a.z - b, a.w - b);                                              \
  }                                                                                            \
  SPARKIUM_HD inline T operator-(float a, const T &b) {                                        \
    return T(a - b.x, a - b.y, a - b.z, a - b.w);                                              \
  }                                                                                            \
  SPARKIUM_HD inline T operator*(const T &a, float b) {                                        \
    return T(a.x * b, a.y * b, a.z * b, a.w * b);                                              \
  }                                                                                            \
  SPARKIUM_HD inline T operator*(float a, const T &b) {                                        \
    return T(a * b.x, a * b.y, a * b.z, a * b.w);                                              \
  }                                                                                            \
  SPARKIUM_HD inline T operator/(const T &a, float b) {                                        \
    return T(a.x / b, a.y / b, a.z / b, a.w / b);                                              \
  }                                                                                            \
  SPARKIUM_HD inline T operator/(float a, const T &b) {                                        \
    return T(a / b.x, a / b.y, a / b.z, a / b.w);                                              \
  }                                                                                            \
  SPARKIUM_HD inline T &operator+=(T &a, const T &b) {                                         \
    a.x += b.x;                                                                                \
    a.y += b.y;                                                                                \
    a.z += b.z;                                                                                \
    a.w += b.w;                                                                                \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator-=(T &a, const T &b) {                                         \
    a.x -= b.x;                                                                                \
    a.y -= b.y;                                                                                \
    a.z -= b.z;                                                                                \
    a.w -= b.w;                                                                                \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator*=(T &a, const T &b) {                                         \
    a.x *= b.x;                                                                                \
    a.y *= b.y;                                                                                \
    a.z *= b.z;                                                                                \
    a.w *= b.w;                                                                                \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator/=(T &a, const T &b) {                                         \
    a.x /= b.x;                                                                                \
    a.y /= b.y;                                                                                \
    a.z /= b.z;                                                                                \
    a.w /= b.w;                                                                                \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator+=(T &a, float b) {                                            \
    a.x += b;                                                                                  \
    a.y += b;                                                                                  \
    a.z += b;                                                                                  \
    a.w += b;                                                                                  \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator-=(T &a, float b) {                                            \
    a.x -= b;                                                                                  \
    a.y -= b;                                                                                  \
    a.z -= b;                                                                                  \
    a.w -= b;                                                                                  \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator*=(T &a, float b) {                                            \
    a.x *= b;                                                                                  \
    a.y *= b;                                                                                  \
    a.z *= b;                                                                                  \
    a.w *= b;                                                                                  \
    return a;                                                                                  \
  }                                                                                            \
  SPARKIUM_HD inline T &operator/=(T &a, float b) {                                            \
    a.x /= b;                                                                                  \
    a.y /= b;                                                                                  \
    a.z /= b;                                                                                  \
    a.w /= b;                                                                                  \
    return a;                                                                                  \
  }

SPARKIUM_DEFINE_VEC2_OPERATORS(float2)
SPARKIUM_DEFINE_VEC3_OPERATORS(float3)
SPARKIUM_DEFINE_VEC4_OPERATORS(float4)

#undef SPARKIUM_DEFINE_VEC2_OPERATORS
#undef SPARKIUM_DEFINE_VEC3_OPERATORS
#undef SPARKIUM_DEFINE_VEC4_OPERATORS

// ---------------------------------------------------------------------------
// Component-wise intrinsics
// ---------------------------------------------------------------------------

SPARKIUM_HD inline float2 abs(const float2 &a) {
  return float2(fabsf(a.x), fabsf(a.y));
}
SPARKIUM_HD inline float3 abs(const float3 &a) {
  return float3(fabsf(a.x), fabsf(a.y), fabsf(a.z));
}
SPARKIUM_HD inline float4 abs(const float4 &a) {
  return float4(fabsf(a.x), fabsf(a.y), fabsf(a.z), fabsf(a.w));
}

SPARKIUM_HD inline float2 min(const float2 &a, const float2 &b) {
  return float2(fminf(a.x, b.x), fminf(a.y, b.y));
}
SPARKIUM_HD inline float3 min(const float3 &a, const float3 &b) {
  return float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
SPARKIUM_HD inline float4 min(const float4 &a, const float4 &b) {
  return float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}
SPARKIUM_HD inline float2 min(const float2 &a, float b) {
  return float2(fminf(a.x, b), fminf(a.y, b));
}
SPARKIUM_HD inline float3 min(const float3 &a, float b) {
  return float3(fminf(a.x, b), fminf(a.y, b), fminf(a.z, b));
}
SPARKIUM_HD inline float4 min(const float4 &a, float b) {
  return float4(fminf(a.x, b), fminf(a.y, b), fminf(a.z, b), fminf(a.w, b));
}

SPARKIUM_HD inline float2 max(const float2 &a, const float2 &b) {
  return float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}
SPARKIUM_HD inline float3 max(const float3 &a, const float3 &b) {
  return float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
SPARKIUM_HD inline float4 max(const float4 &a, const float4 &b) {
  return float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}
SPARKIUM_HD inline float2 max(const float2 &a, float b) {
  return float2(fmaxf(a.x, b), fmaxf(a.y, b));
}
SPARKIUM_HD inline float3 max(const float3 &a, float b) {
  return float3(fmaxf(a.x, b), fmaxf(a.y, b), fmaxf(a.z, b));
}
SPARKIUM_HD inline float4 max(const float4 &a, float b) {
  return float4(fmaxf(a.x, b), fmaxf(a.y, b), fmaxf(a.z, b), fmaxf(a.w, b));
}

SPARKIUM_HD inline float2 clamp(const float2 &a, float lo, float hi) {
  return min(max(a, lo), hi);
}
SPARKIUM_HD inline float3 clamp(const float3 &a, float lo, float hi) {
  return min(max(a, lo), hi);
}
SPARKIUM_HD inline float4 clamp(const float4 &a, float lo, float hi) {
  return min(max(a, lo), hi);
}
SPARKIUM_HD inline float2 clamp(const float2 &a, const float2 &lo, const float2 &hi) {
  return min(max(a, lo), hi);
}
SPARKIUM_HD inline float3 clamp(const float3 &a, const float3 &lo, const float3 &hi) {
  return min(max(a, lo), hi);
}

SPARKIUM_HD inline float2 saturate(const float2 &a) {
  return clamp(a, 0.0f, 1.0f);
}
SPARKIUM_HD inline float3 saturate(const float3 &a) {
  return clamp(a, 0.0f, 1.0f);
}
SPARKIUM_HD inline float4 saturate(const float4 &a) {
  return clamp(a, 0.0f, 1.0f);
}

SPARKIUM_HD inline float2 lerp(const float2 &a, const float2 &b, const float2 &t) {
  return a + t * (b - a);
}
SPARKIUM_HD inline float3 lerp(const float3 &a, const float3 &b, const float3 &t) {
  return a + t * (b - a);
}
SPARKIUM_HD inline float4 lerp(const float4 &a, const float4 &b, const float4 &t) {
  return a + t * (b - a);
}
SPARKIUM_HD inline float2 lerp(const float2 &a, const float2 &b, float t) {
  return a + t * (b - a);
}
SPARKIUM_HD inline float3 lerp(const float3 &a, const float3 &b, float t) {
  return a + t * (b - a);
}
SPARKIUM_HD inline float4 lerp(const float4 &a, const float4 &b, float t) {
  return a + t * (b - a);
}

SPARKIUM_HD inline float3 rcp(const float3 &a) {
  return float3(1.0f / a.x, 1.0f / a.y, 1.0f / a.z);
}

SPARKIUM_HD inline float2 step(const float2 &edge, const float2 &a) {
  return float2(a.x >= edge.x ? 1.0f : 0.0f, a.y >= edge.y ? 1.0f : 0.0f);
}
SPARKIUM_HD inline float3 step(const float3 &edge, const float3 &a) {
  return float3(a.x >= edge.x ? 1.0f : 0.0f, a.y >= edge.y ? 1.0f : 0.0f, a.z >= edge.z ? 1.0f : 0.0f);
}
SPARKIUM_HD inline float4 step(const float4 &edge, const float4 &a) {
  return float4(a.x >= edge.x ? 1.0f : 0.0f, a.y >= edge.y ? 1.0f : 0.0f, a.z >= edge.z ? 1.0f : 0.0f,
                a.w >= edge.w ? 1.0f : 0.0f);
}
// HLSL's scalar step/frac must be declared with the vector overloads above,
// because they are used by them (e.g. `frac(a.x)` inside `frac(float2)`).
SPARKIUM_HD inline float step(float edge, float a) {
  return a >= edge ? 1.0f : 0.0f;
}

SPARKIUM_HD inline float2 floor(const float2 &a) {
  return float2(floorf(a.x), floorf(a.y));
}
SPARKIUM_HD inline float3 floor(const float3 &a) {
  return float3(floorf(a.x), floorf(a.y), floorf(a.z));
}
SPARKIUM_HD inline float3 ceil(const float3 &a) {
  return float3(ceilf(a.x), ceilf(a.y), ceilf(a.z));
}
SPARKIUM_HD inline float frac(float a) {
  return a - floorf(a);
}
SPARKIUM_HD inline float2 frac(const float2 &a) {
  return float2(frac(a.x), frac(a.y));
}
SPARKIUM_HD inline float3 frac(const float3 &a) {
  return float3(frac(a.x), frac(a.y), frac(a.z));
}

SPARKIUM_HD inline float2 sqrt(const float2 &a) {
  return float2(sqrtf(a.x), sqrtf(a.y));
}
SPARKIUM_HD inline float3 sqrt(const float3 &a) {
  return float3(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z));
}
SPARKIUM_HD inline float4 sqrt(const float4 &a) {
  return float4(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z), sqrtf(a.w));
}
SPARKIUM_HD inline float3 rsqrt(const float3 &a) {
  return float3(1.0f / sqrtf(a.x), 1.0f / sqrtf(a.y), 1.0f / sqrtf(a.z));
}

SPARKIUM_HD inline float2 exp(const float2 &a) {
  return float2(expf(a.x), expf(a.y));
}
SPARKIUM_HD inline float3 exp(const float3 &a) {
  return float3(expf(a.x), expf(a.y), expf(a.z));
}
SPARKIUM_HD inline float3 exp2(const float3 &a) {
  return float3(exp2f(a.x), exp2f(a.y), exp2f(a.z));
}
SPARKIUM_HD inline float2 log(const float2 &a) {
  return float2(logf(a.x), logf(a.y));
}
SPARKIUM_HD inline float3 log(const float3 &a) {
  return float3(logf(a.x), logf(a.y), logf(a.z));
}
SPARKIUM_HD inline float3 log2(const float3 &a) {
  return float3(log2f(a.x), log2f(a.y), log2f(a.z));
}

SPARKIUM_HD inline float2 sin(const float2 &a) {
  return float2(sinf(a.x), sinf(a.y));
}
SPARKIUM_HD inline float3 sin(const float3 &a) {
  return float3(sinf(a.x), sinf(a.y), sinf(a.z));
}
SPARKIUM_HD inline float2 cos(const float2 &a) {
  return float2(cosf(a.x), cosf(a.y));
}
SPARKIUM_HD inline float3 cos(const float3 &a) {
  return float3(cosf(a.x), cosf(a.y), cosf(a.z));
}

SPARKIUM_HD inline float3 pow(const float3 &a, const float3 &b) {
  return float3(powf(a.x, b.x), powf(a.y, b.y), powf(a.z, b.z));
}
SPARKIUM_HD inline float3 pow(const float3 &a, float b) {
  return float3(powf(a.x, b), powf(a.y, b), powf(a.z, b));
}

SPARKIUM_HD inline float3 sign(const float3 &a) {
  return float3(a.x > 0.0f ? 1.0f : (a.x < 0.0f ? -1.0f : 0.0f), a.y > 0.0f ? 1.0f : (a.y < 0.0f ? -1.0f : 0.0f),
                a.z > 0.0f ? 1.0f : (a.z < 0.0f ? -1.0f : 0.0f));
}

SPARKIUM_HD inline float dot(const float2 &a, const float2 &b) {
  return a.x * b.x + a.y * b.y;
}
SPARKIUM_HD inline float dot(const float3 &a, const float3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
SPARKIUM_HD inline float dot(const float4 &a, const float4 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
SPARKIUM_HD inline float4 dot4(const float4 &a, const float4 &b) {
  return float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

SPARKIUM_HD inline float3 cross(const float3 &a, const float3 &b) {
  return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

SPARKIUM_HD inline float length(const float2 &a) {
  return sqrtf(dot(a, a));
}
SPARKIUM_HD inline float length(const float3 &a) {
  return sqrtf(dot(a, a));
}
SPARKIUM_HD inline float3 normalize(const float3 &a) {
  const float len = length(a);
  return len > 0.0f ? a / len : a;
}
SPARKIUM_HD inline float2 normalize(const float2 &a) {
  const float len = length(a);
  return len > 0.0f ? a / len : a;
}

SPARKIUM_HD inline float3 reflect(const float3 &i, const float3 &n) {
  return i - 2.0f * dot(n, i) * n;
}

SPARKIUM_HD inline bool all_positive(const float3 &a) {
  return a.x > 0.0f && a.y > 0.0f && a.z > 0.0f;
}
SPARKIUM_HD inline bool all_finite(const float3 &a) {
  return !isnanf(a.x) && !isnanf(a.y) && !isnanf(a.z) && fabsf(a.x) < 3.402823466e38f &&
         fabsf(a.y) < 3.402823466e38f && fabsf(a.z) < 3.402823466e38f;
}
SPARKIUM_HD inline float max_component(const float3 &a) {
  return fmaxf(a.x, fmaxf(a.y, a.z));
}
SPARKIUM_HD inline float3 make_float3(float v) {
  return float3(v, v, v);
}
SPARKIUM_HD inline float3 make_float3(float x, float y, float z) {
  return float3(x, y, z);
}

// ---------------------------------------------------------------------------
// Matrices (rows are stored explicitly, matching HLSL row semantics)
// ---------------------------------------------------------------------------

struct float3x3 {
  float3 r0, r1, r2;
  SPARKIUM_HD float3x3() {
  }
  SPARKIUM_HD float3x3(const float3 &a, const float3 &b, const float3 &c) : r0(a), r1(b), r2(c) {
  }
};

struct float3x4 {
  float4 r0, r1, r2;
  SPARKIUM_HD float3x4() {
  }
  SPARKIUM_HD float3x4(const float4 &a, const float4 &b, const float4 &c) : r0(a), r1(b), r2(c) {
  }
};

struct float4x4 {
  float4 r0, r1, r2, r3;
  SPARKIUM_HD float4x4() {
  }
  SPARKIUM_HD float4x4(const float4 &a, const float4 &b, const float4 &c, const float4 &d)
      : r0(a), r1(b), r2(c), r3(d) {
  }
};

SPARKIUM_HD inline float4 mul(const float4x4 &m, const float4 &v) {
  return float4(dot(m.r0, v), dot(m.r1, v), dot(m.r2, v), dot(m.r3, v));
}

SPARKIUM_HD inline float3 mul(const float3x4 &m, const float4 &v) {
  return float3(dot(m.r0, v), dot(m.r1, v), dot(m.r2, v));
}

SPARKIUM_HD inline float3 mul(const float3x3 &m, const float3 &v) {
  return float3(dot(m.r0, v), dot(m.r1, v), dot(m.r2, v));
}

// HLSL `mul(vector, matrix)` treats the argument as a row vector.
SPARKIUM_HD inline float3 mul(const float3 &v, const float3x3 &m) {
  return m.r0 * v.x + m.r1 * v.y + m.r2 * v.z;
}

SPARKIUM_HD inline float3x3 transpose(const float3x3 &m) {
  return float3x3(float3(m.r0.x, m.r1.x, m.r2.x), float3(m.r0.y, m.r1.y, m.r2.y),
                  float3(m.r0.z, m.r1.z, m.r2.z));
}


// ---------------------------------------------------------------------------
// Integer vectors (used by the byte-address buffer ports)
// ---------------------------------------------------------------------------

struct uint2 {
  uint32_t x{0}, y{0};
  SPARKIUM_HD uint2() {
  }
  SPARKIUM_HD uint2(uint32_t v) : x(v), y(v) {
  }
  SPARKIUM_HD uint2(uint32_t x, uint32_t y) : x(x), y(y) {
  }
  SPARKIUM_HD uint32_t operator[](uint32_t i) const {
    return i == 0 ? x : y;
  }
};

struct uint3 {
  uint32_t x{0}, y{0}, z{0};
  SPARKIUM_HD uint3() {
  }
  SPARKIUM_HD uint3(uint32_t v) : x(v), y(v), z(v) {
  }
  SPARKIUM_HD uint3(uint32_t x, uint32_t y, uint32_t z) : x(x), y(y), z(z) {
  }
  SPARKIUM_HD uint32_t operator[](uint32_t i) const {
    return i == 0 ? x : (i == 1 ? y : z);
  }
};

struct uint4 {
  uint32_t x{0}, y{0}, z{0}, w{0};
  SPARKIUM_HD uint4() {
  }
  SPARKIUM_HD uint4(uint32_t v) : x(v), y(v), z(v), w(v) {
  }
  SPARKIUM_HD uint4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) : x(x), y(y), z(z), w(w) {
  }
  SPARKIUM_HD uint4(const uint3 &xyz, uint32_t w) : x(xyz.x), y(xyz.y), z(xyz.z), w(w) {
  }
  SPARKIUM_HD uint32_t operator[](uint32_t i) const {
    return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w));
  }
};

SPARKIUM_HD inline float3 asfloat(const uint3 &v) {
  return float3(asfloat(v.x), asfloat(v.y), asfloat(v.z));
}
SPARKIUM_HD inline float2 asfloat(const uint2 &v) {
  return float2(asfloat(v.x), asfloat(v.y));
}
SPARKIUM_HD inline float4 asfloat(const uint4 &v) {
  return float4(asfloat(v.x), asfloat(v.y), asfloat(v.z), asfloat(v.w));
}
SPARKIUM_HD inline uint3 asuint(const float3 &v) {
  return uint3(asuint(v.x), asuint(v.y), asuint(v.z));
}
SPARKIUM_HD inline uint2 asuint(const float2 &v) {
  return uint2(asuint(v.x), asuint(v.y));
}
SPARKIUM_HD inline uint4 asuint(const float4 &v) {
  return uint4(asuint(v.x), asuint(v.y), asuint(v.z), asuint(v.w));
}

// ---------------------------------------------------------------------------
// Scalar intrinsics that HLSL spells like the vector overloads above.
//
// Functions that also exist in <math.h> (sqrt, pow, exp, ...) are intentionally
// NOT redeclared here: the transcription calls the C functions directly, which
// keeps overload resolution unambiguous without changing the arithmetic.
// ---------------------------------------------------------------------------

SPARKIUM_HD inline float min(float a, float b) {
  return a < b ? a : b;
}
SPARKIUM_HD inline float max(float a, float b) {
  return a > b ? a : b;
}
SPARKIUM_HD inline float saturate(float a) {
  return saturatef(a);
}
SPARKIUM_HD inline float clamp(float a, float lo, float hi) {
  return clampf(a, lo, hi);
}
SPARKIUM_HD inline float lerp(float a, float b, float t) {
  return a + (b - a) * t;
}
SPARKIUM_HD inline float smoothstep(float e0, float e1, float a) {
  float t = saturatef((a - e0) / (e1 - e0));
  return t * t * (3.0f - 2.0f * t);
}
SPARKIUM_HD inline float sign(float a) {
  return a > 0.0f ? 1.0f : (a < 0.0f ? -1.0f : 0.0f);
}
SPARKIUM_HD inline float mad(float a, float b, float c) {
  return a * b + c;
}
SPARKIUM_HD inline float pows(float a, float b) {
  return powf(a, b);
}
SPARKIUM_HD inline float exps(float a) {
  return expf(a);
}
SPARKIUM_HD inline float log2s(float a) {
  return log2f(a);
}
SPARKIUM_HD inline float sqrts(float a) {
  return sqrtf(a);
}
SPARKIUM_HD inline float sinf_(float a) {
  return sinf(a);
}
SPARKIUM_HD inline float cosf_(float a) {
  return cosf(a);
}
SPARKIUM_HD inline float floorf_(float a) {
  return floorf(a);
}

}  // namespace sparkium::device
