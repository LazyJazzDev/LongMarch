// HLSL vector types for the Sparkium CPU backend.
//
// The CPU path tracer compiles the same HLSL sources the GPU backend does, so
// this header reproduces the HLSL semantics those sources rely on rather than
// merely supplying something that looks similar. Two properties matter most:
//
//   * Vectors are tightly packed: float3 occupies 12 bytes. The shaders
//     reinterpret buffers through these types, and the GPU-side layouts are
//     size-asserted in software_pipeline.cpp (SoftwareNode is 32 bytes,
//     GPUInstance is 112). Anything with SIMD alignment would break them, which
//     is also why glm is unusable here: on ARM its vectors are 16-byte aligned,
//     and its swizzle operators only exist under MSVC language extensions.
//   * Swizzles work as both rvalues and lvalues, including permuting ones such
//     as `.wz`.
//
// A swizzle is a union member aliasing the vector's own storage. The proxy
// keeps a full-size array so a permuting swizzle can index the elements it
// names, then presents only as many components as it was declared with.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace sparkium::cpu::hlsl {

struct Float2;
struct Float3;
struct Float4;
struct Uint2;
struct Uint3;
struct Uint4;
struct Int2;
struct Int3;
struct Int4;
struct Bool2;
struct Bool3;
struct Bool4;

template <class T>
struct Traits {
  static constexpr bool is_vector = false;
  static constexpr bool is_matrix = false;
  static constexpr int components = 0;
  using value_type = T;
};

template <class T>
inline constexpr bool is_vector_v = Traits<T>::is_vector;
template <class T>
inline constexpr bool is_matrix_v = Traits<T>::is_matrix;
template <class T>
inline constexpr bool is_scalar_v = !is_vector_v<T> && !is_matrix_v<T>;
template <class T>
using ValueTypeOf = typename Traits<T>::value_type;
template <class T>
inline constexpr int components_v = Traits<T>::components;

// `Vec` is the vector the swizzle presents. `Storage` matches the owning
// vector's component count so the union member aliases its storage exactly.
template <class Vec, class T, int Storage, int I0, int I1 = -1, int I2 = -1, int I3 = -1>
struct Swz {
  T data[Storage];

  static constexpr int count = I3 >= 0 ? 4 : (I2 >= 0 ? 3 : (I1 >= 0 ? 2 : 1));
  static constexpr int index[4] = {I0, I1 < 0 ? 0 : I1, I2 < 0 ? 0 : I2, I3 < 0 ? 0 : I3};

  template <class V>
  void Fill(V &out) const {
    for (int i = 0; i < count; ++i)
      out[i] = static_cast<ValueTypeOf<V>>(data[index[i]]);
  }

  operator Vec() const {
    Vec out;
    Fill(out);
    return out;
  }

  // Indexing a swizzle yields the i-th component it presents, not the i-th
  // component of the underlying vector.
  T &operator[](int i) {
    return data[index[i]];
  }
  const T &operator[](int i) const {
    return data[index[i]];
  }

  Swz &operator=(T value) {
    for (int i = 0; i < count; ++i)
      data[index[i]] = value;
    return *this;
  }
  Swz &operator=(const Vec &value) {
    for (int i = 0; i < count; ++i)
      data[index[i]] = static_cast<T>(value[i]);
    return *this;
  }

  // Compound assignment rereads the current value, which matters for the
  // overlapping swizzles the shaders use (`accum_color.rgb *= x`).
  template <char Op, class V>
  void Compound(const V &value) {
    Vec current;
    Fill(current);
    for (int i = 0; i < count; ++i) {
      const T lhs = static_cast<T>(current[i]);
      const T rhs = static_cast<T>(value[i]);
      if constexpr (Op == '*')
        data[index[i]] = static_cast<T>(lhs * rhs);
      else if constexpr (Op == '+')
        data[index[i]] = static_cast<T>(lhs + rhs);
      else if constexpr (Op == '-')
        data[index[i]] = static_cast<T>(lhs - rhs);
      else
        data[index[i]] = static_cast<T>(lhs / rhs);
    }
  }
  Swz &operator*=(const Vec &value) {
    Compound<'*'>(value);
    return *this;
  }
  Swz &operator+=(const Vec &value) {
    Compound<'+'>(value);
    return *this;
  }
  Swz &operator-=(const Vec &value) {
    Compound<'-'>(value);
    return *this;
  }
  Swz &operator/=(const Vec &value) {
    Compound<'/'>(value);
    return *this;
  }
  Swz &operator*=(T value) {
    for (int i = 0; i < count; ++i)
      data[index[i]] = static_cast<T>(data[index[i]] * value);
    return *this;
  }
  Swz &operator/=(T value) {
    for (int i = 0; i < count; ++i)
      data[index[i]] = static_cast<T>(data[index[i]] / value);
    return *this;
  }
};

// A swizzle behaves as a value of its own vector type, so that constructs such
// as `asfloat(payload.high.yz)` and `normalize(v.xyz)` work without the
// shaders having to materialise the intermediate vector.
template <class Vec, class T, int Storage, int I0, int I1, int I2, int I3>
struct Traits<Swz<Vec, T, Storage, I0, I1, I2, I3>> {
  static constexpr bool is_vector = true;
  static constexpr bool is_matrix = false;
  static constexpr int components = Swz<Vec, T, Storage, I0, I1, I2, I3>::count;
  using value_type = T;
};

#define SPARKIUM_CPU_TRAITS(NAME, T, N)        \
  template <>                                  \
  struct Traits<NAME> {                        \
    static constexpr bool is_vector = true;    \
    static constexpr bool is_matrix = false;   \
    static constexpr int components = N;       \
    using value_type = T;                      \
  }

#define SPARKIUM_CPU_ACCESSORS(T, V) \
  T &operator[](int i) {             \
    return data[i];                  \
  }                                  \
  const T &operator[](int i) const { \
    return data[i];                  \
  }

// Converts from any vector with at least N components, which is what HLSL does
// when a wider vector is assigned to a narrower one.
#define SPARKIUM_CPU_NARROWING_CTOR(NAME, T, N)                                              \
  template <class V,                                                                         \
            std::enable_if_t<is_vector_v<V> && !std::is_same_v<V, NAME> && components_v<V> >= N, \
                             int> = 0>                                                       \
  NAME(const V &value) {                                                                     \
    for (int i = 0; i < N; ++i)                                                              \
      data[i] = static_cast<T>(value[i]);                                                    \
  }

#define SPARKIUM_CPU_SAME_WIDTH_CTOR(NAME, T, N)                                              \
  template <class V,                                                                          \
            std::enable_if_t<is_vector_v<V> && !std::is_same_v<V, NAME> && components_v<V> == N, \
                             int> = 0>                                                        \
  NAME(const V &value) {                                                                      \
    for (int i = 0; i < N; ++i)                                                               \
      data[i] = static_cast<T>(value[i]);                                                     \
  }

struct Float2 {
  union {
    float data[2];
    struct {
      float x, y;
    };
    struct {
      float r, g;
    };
    Swz<Float2, float, 2, 0, 1> xy;
  };
  Float2() : data{} {
  }
  Float2(float value) : data{value, value} {
  }
  Float2(float a, float b) : data{a, b} {
  }
  template <class V, std::enable_if_t<is_vector_v<V> && !std::is_same_v<V, Float2> && components_v<V> >= 2,
                                      int> = 0>
  Float2(const V &value) {
    for (int i = 0; i < 2; ++i)
      data[i] = static_cast<float>(value[i]);
  }
  SPARKIUM_CPU_ACCESSORS(float, Float2)
};

struct Float3 {
  union {
    float data[3];
    struct {
      float x, y, z;
    };
    struct {
      float r, g, b;
    };
    Swz<Float2, float, 3, 0, 1> xy;
    Swz<Float2, float, 3, 1, 2> yz;
    // The HSV conversion in the generated preamble permutes colour channels on
    // a float3.
    Swz<Float2, float, 3, 2, 1> bg;
    Swz<Float2, float, 3, 1, 2> gb;
    Swz<Float3, float, 3, 0, 1, 2> xyz;
    Swz<Float3, float, 3, 0, 0, 0> xxx;
    Swz<Float4, float, 3, 0, 0, 0, 0> xxxx;
  };
  Float3() : data{} {
  }
  Float3(float value) : data{value, value, value} {
  }
  Float3(float a, float b, float c) : data{a, b, c} {
  }
  template <class V, std::enable_if_t<is_vector_v<V> && components_v<V> == 2, int> = 0>
  Float3(float a, const V &v) : data{a, static_cast<float>(v[0]), static_cast<float>(v[1])} {
  }
  template <class V, std::enable_if_t<is_vector_v<V> && !std::is_same_v<V, Float3> && components_v<V> >= 3,
                                      int> = 0>
  Float3(const V &value) {
    for (int i = 0; i < 3; ++i)
      data[i] = static_cast<float>(value[i]);
  }
  template <class V, std::enable_if_t<is_vector_v<V> && components_v<V> == 2, int> = 0>
  Float3(const V &v, float c) : data{static_cast<float>(v[0]), static_cast<float>(v[1]), c} {
  }
  SPARKIUM_CPU_ACCESSORS(float, Float3)
};

struct Float4 {
  union {
    float data[4];
    struct {
      float x, y, z, w;
    };
    struct {
      float r, g, b, a;
    };
    Swz<Float2, float, 4, 0, 1> xy;
    Swz<Float3, float, 4, 0, 1, 2> xyz;
    Swz<Float3, float, 4, 0, 1, 2> rgb;
    Swz<Float2, float, 4, 3, 2> wz;
    Swz<Float2, float, 4, 2, 1> bg;
    Swz<Float2, float, 4, 1, 2> gb;
    Swz<Float3, float, 4, 0, 1, 3> xyw;
    Swz<Float3, float, 4, 1, 2, 0> yzx;
  };
  Float4() : data{} {
  }
  Float4(float value) : data{value, value, value, value} {
  }
  Float4(float a, float b, float c, float d) : data{a, b, c, d} {
  }
  // HLSL constructors concatenate whatever they are given, so a leading scalar
  // followed by a vector is as valid as the other way round.
  template <class V, std::enable_if_t<is_vector_v<V> && components_v<V> == 3, int> = 0>
  Float4(float a, const V &v)
      : data{a, static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2])} {
  }
  template <class V, std::enable_if_t<is_vector_v<V> && components_v<V> == 2, int> = 0>
  Float4(float a, const V &v, float b) : data{a, static_cast<float>(v[0]), static_cast<float>(v[1]), b} {
  }
  template <class V, std::enable_if_t<is_vector_v<V> && !std::is_same_v<V, Float4> && components_v<V> >= 4,
                                      int> = 0>
  Float4(const V &value) {
    for (int i = 0; i < 4; ++i)
      data[i] = static_cast<float>(value[i]);
  }
  template <class V, std::enable_if_t<is_vector_v<V> && components_v<V> == 3, int> = 0>
  Float4(const V &v, float d)
      : data{static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2]), d} {
  }
  template <class V, std::enable_if_t<is_vector_v<V> && components_v<V> == 2, int> = 0>
  Float4(const V &v, float c, float d)
      : data{static_cast<float>(v[0]), static_cast<float>(v[1]), c, d} {
  }
  template <class A, class B, std::enable_if_t<is_vector_v<A> && components_v<A> == 2 && is_vector_v<B> &&
                                                   components_v<B> == 2,
                                               int> = 0>
  Float4(const A &a, const B &b)
      : data{static_cast<float>(a[0]), static_cast<float>(a[1]), static_cast<float>(b[0]),
             static_cast<float>(b[1])} {
  }
  SPARKIUM_CPU_ACCESSORS(float, Float4)
};

struct Uint2 {
  union {
    uint32_t data[2];
    struct {
      uint32_t x, y;
    };
    Swz<Uint2, uint32_t, 2, 0, 1> xy;
  };
  Uint2() : data{} {
  }
  Uint2(uint32_t value) : data{value, value} {
  }
  Uint2(uint32_t a, uint32_t b) : data{a, b} {
  }
  template <class V, std::enable_if_t<is_vector_v<V> && !std::is_same_v<V, Uint2> && components_v<V> >= 2,
                                      int> = 0>
  Uint2(const V &value) {
    for (int i = 0; i < 2; ++i)
      data[i] = static_cast<uint32_t>(value[i]);
  }
  SPARKIUM_CPU_ACCESSORS(uint32_t, Uint2)
};

struct Uint3 {
  union {
    uint32_t data[3];
    struct {
      uint32_t x, y, z;
    };
    Swz<Uint2, uint32_t, 3, 0, 1> xy;
    Swz<Uint3, uint32_t, 3, 0, 1, 2> xyz;
  };
  Uint3() : data{} {
  }
  Uint3(uint32_t value) : data{value, value, value} {
  }
  Uint3(uint32_t a, uint32_t b, uint32_t c) : data{a, b, c} {
  }
  template <class V, std::enable_if_t<is_vector_v<V> && !std::is_same_v<V, Uint3> && components_v<V> >= 3,
                                      int> = 0>
  Uint3(const V &value) {
    for (int i = 0; i < 3; ++i)
      data[i] = static_cast<uint32_t>(value[i]);
  }
  SPARKIUM_CPU_ACCESSORS(uint32_t, Uint3)
};

struct Uint4 {
  union {
    uint32_t data[4];
    struct {
      uint32_t x, y, z, w;
    };
    Swz<Uint2, uint32_t, 4, 1, 2> yz;
    Swz<Uint3, uint32_t, 4, 0, 1, 2> xyz;
    Swz<Uint4, uint32_t, 4, 3, 3, 3, 3> wwww;
  };
  Uint4() : data{} {
  }
  Uint4(uint32_t value) : data{value, value, value, value} {
  }
  Uint4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) : data{a, b, c, d} {
  }
  template <class V, std::enable_if_t<is_vector_v<V> && !std::is_same_v<V, Uint4> && components_v<V> >= 4,
                                      int> = 0>
  Uint4(const V &value) {
    for (int i = 0; i < 4; ++i)
      data[i] = static_cast<uint32_t>(value[i]);
  }
  SPARKIUM_CPU_ACCESSORS(uint32_t, Uint4)
};

struct Int2 {
  union {
    int32_t data[2];
    struct {
      int32_t x, y;
    };
    Swz<Int2, int32_t, 2, 0, 1> xy;
  };
  Int2() : data{} {
  }
  Int2(int32_t value) : data{value, value} {
  }
  Int2(int32_t a, int32_t b) : data{a, b} {
  }
  template <class V, std::enable_if_t<is_vector_v<V> && !std::is_same_v<V, Int2> && components_v<V> >= 2,
                                      int> = 0>
  Int2(const V &value) {
    for (int i = 0; i < 2; ++i)
      data[i] = static_cast<int32_t>(value[i]);
  }
  SPARKIUM_CPU_ACCESSORS(int32_t, Int2)
};

struct Int3 {
  union {
    int32_t data[3];
    struct {
      int32_t x, y, z;
    };
    Swz<Int2, int32_t, 3, 0, 1> xy;
    Swz<Int3, int32_t, 3, 0, 1, 2> xyz;
  };
  Int3() : data{} {
  }
  Int3(int32_t value) : data{value, value, value} {
  }
  Int3(int32_t a, int32_t b, int32_t c) : data{a, b, c} {
  }
  template <class V, std::enable_if_t<is_vector_v<V> && components_v<V> == 2, int> = 0>
  Int3(const V &v, int32_t c) : data{static_cast<int32_t>(v[0]), static_cast<int32_t>(v[1]), c} {
  }
  template <class V, std::enable_if_t<is_vector_v<V> && !std::is_same_v<V, Int3> && components_v<V> >= 3,
                                      int> = 0>
  Int3(const V &value) {
    for (int i = 0; i < 3; ++i)
      data[i] = static_cast<int32_t>(value[i]);
  }
  SPARKIUM_CPU_ACCESSORS(int32_t, Int3)
};

struct Int4 {
  union {
    int32_t data[4];
    struct {
      int32_t x, y, z, w;
    };
    Swz<Int3, int32_t, 4, 0, 1, 2> xyz;
  };
  Int4() : data{} {
  }
  Int4(int32_t value) : data{value, value, value, value} {
  }
  Int4(int32_t a, int32_t b, int32_t c, int32_t d) : data{a, b, c, d} {
  }
  template <class V, std::enable_if_t<is_vector_v<V> && !std::is_same_v<V, Int4> && components_v<V> >= 4,
                                      int> = 0>
  Int4(const V &value) {
    for (int i = 0; i < 4; ++i)
      data[i] = static_cast<int32_t>(value[i]);
  }
  SPARKIUM_CPU_ACCESSORS(int32_t, Int4)
};

// Boolean vectors only ever appear as the result of a comparison, so they need
// no swizzles.
#define SPARKIUM_CPU_DECLARE_BOOL(NAME, N)  \
  struct NAME {                             \
    bool data[N];                           \
    NAME() : data{} {                       \
    }                                       \
    NAME(bool value) {                      \
      for (int i = 0; i < N; ++i)           \
        data[i] = value;                    \
    }                                       \
    bool &operator[](int i) {               \
      return data[i];                       \
    }                                       \
    const bool &operator[](int i) const {   \
      return data[i];                       \
    }                                       \
  };                                        \
  SPARKIUM_CPU_TRAITS(NAME, bool, N)

SPARKIUM_CPU_DECLARE_BOOL(Bool2, 2);
SPARKIUM_CPU_DECLARE_BOOL(Bool3, 3);
SPARKIUM_CPU_DECLARE_BOOL(Bool4, 4);

#undef SPARKIUM_CPU_DECLARE_BOOL

SPARKIUM_CPU_TRAITS(Float2, float, 2);
SPARKIUM_CPU_TRAITS(Float3, float, 3);
SPARKIUM_CPU_TRAITS(Float4, float, 4);
SPARKIUM_CPU_TRAITS(Uint2, uint32_t, 2);
SPARKIUM_CPU_TRAITS(Uint3, uint32_t, 3);
SPARKIUM_CPU_TRAITS(Uint4, uint32_t, 4);
SPARKIUM_CPU_TRAITS(Int2, int32_t, 2);
SPARKIUM_CPU_TRAITS(Int3, int32_t, 3);
SPARKIUM_CPU_TRAITS(Int4, int32_t, 4);

#undef SPARKIUM_CPU_TRAITS
#undef SPARKIUM_CPU_ACCESSORS
#undef SPARKIUM_CPU_NARROWING_CTOR
#undef SPARKIUM_CPU_SAME_WIDTH_CTOR

// Maps a (width, element type) pair back to the vector type that carries it,
// which is how intrinsics such as asfloat and transpose name their results.
template <int N, class T>
struct VecOf;
template <>
struct VecOf<2, float> {
  using type = Float2;
};
template <>
struct VecOf<3, float> {
  using type = Float3;
};
template <>
struct VecOf<4, float> {
  using type = Float4;
};
template <>
struct VecOf<2, uint32_t> {
  using type = Uint2;
};
template <>
struct VecOf<3, uint32_t> {
  using type = Uint3;
};
template <>
struct VecOf<4, uint32_t> {
  using type = Uint4;
};
template <>
struct VecOf<2, int32_t> {
  using type = Int2;
};
template <>
struct VecOf<3, int32_t> {
  using type = Int3;
};
template <>
struct VecOf<4, int32_t> {
  using type = Int4;
};
template <int N, class T>
using VecOfT = typename VecOf<N, T>::type;

// The names the shaders actually spell.
using float2 = Float2;
using float3 = Float3;
using float4 = Float4;
using uint2 = Uint2;
using uint3 = Uint3;
using uint4 = Uint4;
using int2 = Int2;
using int3 = Int3;
using int4 = Int4;
using bool2 = Bool2;
using bool3 = Bool3;
using bool4 = Bool4;

}  // namespace sparkium::cpu::hlsl
