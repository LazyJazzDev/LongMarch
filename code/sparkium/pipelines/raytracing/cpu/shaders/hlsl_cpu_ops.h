// HLSL operators and intrinsics for the Sparkium CPU backend.
//
// Only the operations the shaders actually use are provided. The arithmetic
// operators follow HLSL's elementwise rules: a scalar operand broadcasts, and
// the result takes the left operand's element type (HLSL promotes to the wider
// type, but the shaders never mix element types in a way where that differs).
#pragma once

#include <cmath>
#include <cstring>
#include <utility>

#include "sparkium/pipelines/raytracing/cpu/shaders/hlsl_cpu_vector.h"

namespace sparkium::cpu::hlsl {

template <int N>
struct BoolOf;
template <>
struct BoolOf<2> {
  using type = Bool2;
};
template <>
struct BoolOf<3> {
  using type = Bool3;
};
template <>
struct BoolOf<4> {
  using type = Bool4;
};
template <int N>
using BoolOfT = typename BoolOf<N>::type;

template <class A, class B>
inline constexpr bool same_shape_v = is_vector_v<A> && components_v<A> == components_v<B>;

// Arithmetic -----------------------------------------------------------------

#define SPARKIUM_CPU_VECTOR_SCALAR_OP(OP)                                                          \
  template <class A, class B, std::enable_if_t<is_vector_v<A> && is_scalar_v<B>, int> = 0>         \
  A operator OP(const A &a, B b) {                                                                 \
    A result;                                                                                      \
    for (int i = 0; i < components_v<A>; ++i)                                                      \
      result[i] = static_cast<ValueTypeOf<A>>(a[i] OP b);                                          \
    return result;                                                                                 \
  }                                                                                                \
  template <class A, class B, std::enable_if_t<is_scalar_v<A> && is_vector_v<B>, int> = 0>         \
  B operator OP(A a, const B &b) {                                                                 \
    B result;                                                                                      \
    for (int i = 0; i < components_v<B>; ++i)                                                      \
      result[i] = static_cast<ValueTypeOf<B>>(a OP b[i]);                                          \
    return result;                                                                                 \
  }                                                                                                \
  template <class A, class B, std::enable_if_t<is_vector_v<A> && is_vector_v<B>, int> = 0>         \
  A operator OP(const A &a, const B &b) {                                                          \
    static_assert(components_v<A> == components_v<B>, "vector size mismatch");                     \
    A result;                                                                                      \
    for (int i = 0; i < components_v<A>; ++i)                                                      \
      result[i] = static_cast<ValueTypeOf<A>>(a[i] OP b[i]);                                       \
    return result;                                                                                 \
  }

SPARKIUM_CPU_VECTOR_SCALAR_OP(+)
SPARKIUM_CPU_VECTOR_SCALAR_OP(-)
SPARKIUM_CPU_VECTOR_SCALAR_OP(*)
SPARKIUM_CPU_VECTOR_SCALAR_OP(/)

#undef SPARKIUM_CPU_VECTOR_SCALAR_OP

// Compound assignment on vectors themselves. Swizzle proxies carry their own,
// which is what lets `v.xyz = w` work.
#define SPARKIUM_CPU_VECTOR_COMPOUND(OP)                                                           \
  template <class A, class B, std::enable_if_t<is_vector_v<A> && is_scalar_v<B>, int> = 0>         \
  A &operator OP##=(A &a, B b) {                                                                   \
    for (int i = 0; i < components_v<A>; ++i)                                                      \
      a[i] = static_cast<ValueTypeOf<A>>(a[i] OP b);                                               \
    return a;                                                                                      \
  }                                                                                                \
  template <class A, class B, std::enable_if_t<is_vector_v<A> && is_vector_v<B>, int> = 0>         \
  A &operator OP##=(A &a, const B &b) {                                                            \
    static_assert(components_v<A> == components_v<B>, "vector size mismatch");                     \
    for (int i = 0; i < components_v<A>; ++i)                                                      \
      a[i] = static_cast<ValueTypeOf<A>>(a[i] OP b[i]);                                            \
    return a;                                                                                      \
  }

SPARKIUM_CPU_VECTOR_COMPOUND(+)
SPARKIUM_CPU_VECTOR_COMPOUND(-)
SPARKIUM_CPU_VECTOR_COMPOUND(*)
SPARKIUM_CPU_VECTOR_COMPOUND(/)

#undef SPARKIUM_CPU_VECTOR_COMPOUND

template <class A, std::enable_if_t<is_vector_v<A>, int> = 0>
A operator-(const A &a) {
  A result;
  for (int i = 0; i < components_v<A>; ++i)
    result[i] = static_cast<ValueTypeOf<A>>(-a[i]);
  return result;
}

// Comparisons return a boolean vector, as in HLSL.
#define SPARKIUM_CPU_COMPARE_OP(OP)                                                                \
  template <class A, class B, std::enable_if_t<same_shape_v<A, B>, int> = 0>                       \
  BoolOfT<components_v<A>> operator OP(const A &a, const B &b) {                                   \
    BoolOfT<components_v<A>> result;                                                               \
    for (int i = 0; i < components_v<A>; ++i)                                                      \
      result[i] = a[i] OP b[i];                                                                    \
    return result;                                                                                 \
  }                                                                                                \
  template <class A, class B, std::enable_if_t<is_vector_v<A> && is_scalar_v<B>, int> = 0>         \
  BoolOfT<components_v<A>> operator OP(const A &a, B b) {                                          \
    BoolOfT<components_v<A>> result;                                                               \
    for (int i = 0; i < components_v<A>; ++i)                                                      \
      result[i] = a[i] OP b;                                                                       \
    return result;                                                                                 \
  }                                                                                                \
  template <class A, class B, std::enable_if_t<is_scalar_v<A> && is_vector_v<B>, int> = 0>         \
  BoolOfT<components_v<B>> operator OP(A a, const B &b) {                                          \
    BoolOfT<components_v<B>> result;                                                               \
    for (int i = 0; i < components_v<B>; ++i)                                                      \
      result[i] = a OP b[i];                                                                       \
    return result;                                                                                 \
  }

SPARKIUM_CPU_COMPARE_OP(==)
SPARKIUM_CPU_COMPARE_OP(!=)
SPARKIUM_CPU_COMPARE_OP(<)
SPARKIUM_CPU_COMPARE_OP(>)
SPARKIUM_CPU_COMPARE_OP(<=)
SPARKIUM_CPU_COMPARE_OP(>=)

#undef SPARKIUM_CPU_COMPARE_OP

// Scalar intrinsics ----------------------------------------------------------
//
// The shaders are compiled in a namespace that imports this one, so the names
// declared here are found alongside the C library's. `min`, `max` and `clamp`
// therefore need scalar overloads: without one, an unqualified `max(a, b)` on
// two floats would only see the vector template below and fail to compile
// rather than fall back to a scalar. Everything else (<cmath>'s sin, sqrt,
// floor, abs, ...) already has a global overload and adding one here would make
// the call ambiguous instead.
//
// Scalars keep HLSL's float-preferring promotion, so `max(x, 1.0)` stays a
// float rather than widening to double.
template <class A, class B, std::enable_if_t<std::is_arithmetic_v<A> && std::is_arithmetic_v<B>, int> = 0>
auto PromoteScalars(A a, B b) {
  using Result = std::conditional_t<std::is_floating_point_v<A> || std::is_floating_point_v<B>, float,
                                    std::common_type_t<A, B>>;
  return std::pair<Result, Result>{static_cast<Result>(a), static_cast<Result>(b)};
}

template <class A, class B, std::enable_if_t<std::is_arithmetic_v<A> && std::is_arithmetic_v<B>, int> = 0>
auto min(A a, B b) {
  const auto [x, y] = PromoteScalars(a, b);
  return x < y ? x : y;
}
template <class A, class B, std::enable_if_t<std::is_arithmetic_v<A> && std::is_arithmetic_v<B>, int> = 0>
auto max(A a, B b) {
  const auto [x, y] = PromoteScalars(a, b);
  return x > y ? x : y;
}
template <class V, class B, class C,
          std::enable_if_t<std::is_arithmetic_v<V> && std::is_arithmetic_v<B> && std::is_arithmetic_v<C>,
                           int> = 0>
auto clamp(V value, B low, C high) {
  const auto [x, lo] = PromoteScalars(value, low);
  const auto [ignored, hi] = PromoteScalars(value, high);
  (void)ignored;
  return x < lo ? lo : (x > hi ? hi : x);
}

// Scalar helpers -------------------------------------------------------------

inline float frac(float x) {
  return x - std::floor(x);
}
inline float saturate(float x) {
  return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}
inline float lerp(float a, float b, float t) {
  // HLSL defines this as a + t * (b - a). The algebraically equivalent
  // a + (b - a) * t rounds differently, which shows up as a small systematic
  // difference from the GPU on shaders that interpolate every sample.
  return a + t * (b - a);
}
inline float rsqrt(float x) {
  return 1.0f / std::sqrt(x);
}
inline float rcp(float x) {
  return 1.0f / x;
}
inline float sign(float x) {
  return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
}
inline float step(float edge, float x) {
  return x < edge ? 0.0f : 1.0f;
}
inline float smoothstep(float a, float b, float x) {
  const float t = saturate((x - a) / (b - a));
  return t * t * (3.0f - 2.0f * t);
}
// fmod, exp2, log2, atan2, isfinite and isnan already have scalar overloads in
// the C library, so only their vector forms are declared here.

// Bit reinterpretation, matching HLSL's asfloat/asuint/asint.
inline uint32_t asuint(float x) {
  uint32_t result;
  std::memcpy(&result, &x, sizeof(result));
  return result;
}
inline int32_t asint(float x) {
  int32_t result;
  std::memcpy(&result, &x, sizeof(result));
  return result;
}
inline float asfloat(uint32_t x) {
  float result;
  std::memcpy(&result, &x, sizeof(result));
  return result;
}
inline float asfloat(int32_t x) {
  float result;
  std::memcpy(&result, &x, sizeof(result));
  return result;
}

// Vector intrinsics ----------------------------------------------------------

#define SPARKIUM_CPU_UNARY(NAME, EXPR)                                                             \
  template <class V, std::enable_if_t<is_vector_v<V>, int> = 0>                                    \
  V NAME(const V &v) {                                                                             \
    V result;                                                                                      \
    for (int i = 0; i < components_v<V>; ++i) {                                                    \
      const auto x = v[i];                                                                         \
      result[i] = static_cast<ValueTypeOf<V>>(EXPR);                                               \
    }                                                                                              \
    return result;                                                                                 \
  }

SPARKIUM_CPU_UNARY(frac, frac(x))
SPARKIUM_CPU_UNARY(saturate, saturate(x))
SPARKIUM_CPU_UNARY(sqrt, std::sqrt(x))
SPARKIUM_CPU_UNARY(rsqrt, rsqrt(x))
SPARKIUM_CPU_UNARY(rcp, rcp(x))
SPARKIUM_CPU_UNARY(sign, sign(x))
SPARKIUM_CPU_UNARY(floor, std::floor(x))
SPARKIUM_CPU_UNARY(ceil, std::ceil(x))
SPARKIUM_CPU_UNARY(trunc, std::trunc(x))
SPARKIUM_CPU_UNARY(round, std::round(x))
SPARKIUM_CPU_UNARY(abs, std::abs(x))
SPARKIUM_CPU_UNARY(exp, std::exp(x))
SPARKIUM_CPU_UNARY(exp2, std::exp2(x))
SPARKIUM_CPU_UNARY(log, std::log(x))
SPARKIUM_CPU_UNARY(log2, std::log2(x))
SPARKIUM_CPU_UNARY(sin, std::sin(x))
SPARKIUM_CPU_UNARY(cos, std::cos(x))
SPARKIUM_CPU_UNARY(tan, std::tan(x))
SPARKIUM_CPU_UNARY(asin, std::asin(x))
SPARKIUM_CPU_UNARY(acos, std::acos(x))
SPARKIUM_CPU_UNARY(atan, std::atan(x))
SPARKIUM_CPU_UNARY(erf, std::erf(x))

#undef SPARKIUM_CPU_UNARY

// Vector-to-vector intrinsics take their result element type from the first
// operand, so `pow(vec3, float)` and `max(vec3, float)` behave like HLSL.
#define SPARKIUM_CPU_BINARY(NAME, EXPR)                                                            \
  template <class A, class B, std::enable_if_t<is_vector_v<A> && is_scalar_v<B>, int> = 0>         \
  A NAME(const A &a, B b) {                                                                        \
    A result;                                                                                      \
    for (int i = 0; i < components_v<A>; ++i) {                                                    \
      const auto x = a[i];                                                                         \
      const auto y = b;                                                                            \
      result[i] = static_cast<ValueTypeOf<A>>(EXPR);                                               \
    }                                                                                              \
    return result;                                                                                 \
  }                                                                                                \
  template <class A, class B, std::enable_if_t<is_vector_v<A> && is_vector_v<B>, int> = 0>         \
  A NAME(const A &a, const B &b) {                                                                 \
    static_assert(components_v<A> == components_v<B>, "vector size mismatch");                     \
    A result;                                                                                      \
    for (int i = 0; i < components_v<A>; ++i) {                                                    \
      const auto x = a[i];                                                                         \
      const auto y = b[i];                                                                         \
      result[i] = static_cast<ValueTypeOf<A>>(EXPR);                                               \
    }                                                                                              \
    return result;                                                                                 \
  }                                                                                                \
  template <class A, class B, std::enable_if_t<is_scalar_v<A> && is_vector_v<B>, int> = 0>         \
  B NAME(A a, const B &b) {                                                                        \
    B result;                                                                                      \
    for (int i = 0; i < components_v<B>; ++i) {                                                    \
      const auto x = a;                                                                            \
      const auto y = b[i];                                                                         \
      result[i] = static_cast<ValueTypeOf<B>>(EXPR);                                               \
    }                                                                                              \
    return result;                                                                                 \
  }

SPARKIUM_CPU_BINARY(min, x < y ? x : y)
SPARKIUM_CPU_BINARY(max, x > y ? x : y)
SPARKIUM_CPU_BINARY(pow, std::pow(x, y))
SPARKIUM_CPU_BINARY(step, step(x, y))
SPARKIUM_CPU_BINARY(atan2, std::atan2(x, y))
SPARKIUM_CPU_BINARY(fmod, std::fmod(x, y))

#undef SPARKIUM_CPU_BINARY

template <class A, class B, class C, std::enable_if_t<is_vector_v<A>, int> = 0>
A clamp(const A &value, B low, C high) {
  A result;
  for (int i = 0; i < components_v<A>; ++i) {
    const auto x = value[i];
    result[i] = static_cast<ValueTypeOf<A>>(x < low ? low : (x > high ? high : x));
  }
  return result;
}

template <class A, class B, class C,
          std::enable_if_t<is_vector_v<A> && is_vector_v<B> && is_scalar_v<C>, int> = 0>
A lerp(const A &a, const B &b, C t) {
  static_assert(components_v<A> == components_v<B>, "vector size mismatch");
  A result;
  for (int i = 0; i < components_v<A>; ++i)
    result[i] = static_cast<ValueTypeOf<A>>(a[i] + t * (b[i] - a[i]));
  return result;
}

template <class A, class B, class C,
          std::enable_if_t<is_vector_v<A> && is_vector_v<B> && is_vector_v<C>, int> = 0>
A lerp(const A &a, const B &b, const C &t) {
  static_assert(components_v<A> == components_v<B>, "vector size mismatch");
  A result;
  for (int i = 0; i < components_v<A>; ++i) {
    const auto x = a[i];
    result[i] = static_cast<ValueTypeOf<A>>(x + (b[i] - x) * t[i]);
  }
  return result;
}

template <class A, class B, std::enable_if_t<is_vector_v<A>, int> = 0>
float dot(const A &a, const B &b) {
  static_assert(components_v<A> == components_v<B>, "vector size mismatch");
  float result = 0.0f;
  for (int i = 0; i < components_v<A>; ++i)
    result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
  return result;
}

template <class A, std::enable_if_t<is_vector_v<A>, int> = 0>
float length(const A &a) {
  return std::sqrt(dot(a, a));
}

template <class A, class B, std::enable_if_t<is_vector_v<A>, int> = 0>
float distance(const A &a, const B &b) {
  return length(a - b);
}

template <class A, std::enable_if_t<is_vector_v<A>, int> = 0>
A normalize(const A &a) {
  const float len = length(a);
  A result;
  for (int i = 0; i < components_v<A>; ++i)
    result[i] = static_cast<ValueTypeOf<A>>(a[i] / len);
  return result;
}

template <class A, class B, std::enable_if_t<is_vector_v<A> && components_v<A> == 3, int> = 0>
A cross(const A &a, const B &b) {
  A result;
  result[0] = static_cast<ValueTypeOf<A>>(a[1] * b[2] - a[2] * b[1]);
  result[1] = static_cast<ValueTypeOf<A>>(a[2] * b[0] - a[0] * b[2]);
  result[2] = static_cast<ValueTypeOf<A>>(a[0] * b[1] - a[1] * b[0]);
  return result;
}

template <class A, class B, std::enable_if_t<is_vector_v<A> && components_v<A> == 3, int> = 0>
A reflect(const A &incident, const B &normal) {
  const float scale = 2.0f * dot(incident, normal);
  A result;
  for (int i = 0; i < 3; ++i)
    result[i] = static_cast<ValueTypeOf<A>>(incident[i] - scale * normal[i]);
  return result;
}

template <class A, class B, class C, std::enable_if_t<is_vector_v<A> && components_v<A> == 3, int> = 0>
A refract(const A &incident, const B &normal, C eta) {
  const float cosi = -dot(normal, incident);
  const float k = 1.0f - static_cast<float>(eta) * static_cast<float>(eta) * (1.0f - cosi * cosi);
  A result;
  if (k < 0.0f) {
    for (int i = 0; i < 3; ++i)
      result[i] = 0.0f;
    return result;
  }
  const float scale = static_cast<float>(eta) * cosi - std::sqrt(k);
  for (int i = 0; i < 3; ++i)
    result[i] = static_cast<ValueTypeOf<A>>(static_cast<float>(eta) * incident[i] + scale * normal[i]);
  return result;
}

template <class B, std::enable_if_t<is_vector_v<B>, int> = 0>
bool any(const B &value) {
  for (int i = 0; i < components_v<B>; ++i)
    if (value[i])
      return true;
  return false;
}

template <class B, std::enable_if_t<is_vector_v<B>, int> = 0>
bool all(const B &value) {
  for (int i = 0; i < components_v<B>; ++i)
    if (!value[i])
      return false;
  return true;
}

template <class V, std::enable_if_t<is_vector_v<V>, int> = 0>
BoolOfT<components_v<V>> isfinite(const V &value) {
  BoolOfT<components_v<V>> result;
  for (int i = 0; i < components_v<V>; ++i)
    result[i] = std::isfinite(static_cast<double>(value[i]));
  return result;
}

template <class V, std::enable_if_t<is_vector_v<V>, int> = 0>
BoolOfT<components_v<V>> isnan(const V &value) {
  BoolOfT<components_v<V>> result;
  for (int i = 0; i < components_v<V>; ++i)
    result[i] = std::isnan(static_cast<double>(value[i]));
  return result;
}

// Bit reinterpretation for vectors. The shaders cast every width, including
// the two- and three-component swizzles of a payload word.
template <class V, std::enable_if_t<is_vector_v<V>, int> = 0>
VecOfT<components_v<V>, uint32_t> asuint(const V &value) {
  VecOfT<components_v<V>, uint32_t> result;
  for (int i = 0; i < components_v<V>; ++i)
    result[i] = asuint(static_cast<float>(value[i]));
  return result;
}

template <class V, std::enable_if_t<is_vector_v<V>, int> = 0>
VecOfT<components_v<V>, float> asfloat(const V &value) {
  VecOfT<components_v<V>, float> result;
  for (int i = 0; i < components_v<V>; ++i)
    result[i] = asfloat(static_cast<uint32_t>(value[i]));
  return result;
}

// Stands in for HLSL's `((x).xxxx)`, which the generated shader graphs use to
// broadcast. It works on a scalar as well as a vector: HLSL repeats the first
// component, so `(v).xxxx` and a scalar broadcast are the same operation.
inline Float4 Splat4(float value) {
  return Float4(value, value, value, value);
}
inline Float4 Splat4(int32_t value) {
  return Splat4(static_cast<float>(value));
}
inline Float4 Splat4(uint32_t value) {
  return Splat4(static_cast<float>(value));
}
template <class V, std::enable_if_t<is_vector_v<V>, int> = 0>
Float4 Splat4(const V &value) {
  return Splat4(static_cast<float>(value[0]));
}

// make_floatN appear in the Principled BSDF sources.
inline Float3 make_float3(float x, float y, float z) {
  return Float3(x, y, z);
}
inline Float3 make_float3(float v) {
  return Float3(v, v, v);
}
inline Float2 make_float2(float x, float y) {
  return Float2(x, y);
}
inline Float4 make_float4(float x, float y, float z, float w) {
  return Float4(x, y, z, w);
}

}  // namespace sparkium::cpu::hlsl
