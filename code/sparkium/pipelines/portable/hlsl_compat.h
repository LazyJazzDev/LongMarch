#pragma once
// HLSL-to-C++ compatibility layer. The shared shading kernels in
// code/sparkium/shaders are written in an HLSL subset; this header provides
// vector/matrix/buffer types and intrinsics so the same sources compile as
// host C++ and as CUDA device code. Keep this file free of HLSL code: it is
// included before every generated kernel translation unit.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
// Under nvcc, cuda_runtime.h defines ::float2/::float3/... at global scope.
// Including it here would make every bare `float3` in the transpiled shading
// code ambiguous once `using namespace sparkium_portable;` is in effect
// (sparkium_portable::float3 vs ::float3). The device compilation of this
// header only needs the __device__/__constant__ keywords, which nvcc defines
// implicitly for .cu translation units, so the CUDA runtime header is
// intentionally NOT included here. Host-side CUDA code (kernel_cuda.cu)
// includes <cuda_runtime.h> itself after this header with fully-qualified
// references.
#define SPARKIUM_KERNEL __device__
#else
#define SPARKIUM_KERNEL
#endif

namespace sparkium_portable {

typedef int32_t int32;
typedef uint32_t uint;
typedef uint16_t uint16;

// ---------------------------------------------------------------------------
// Small vector types with HLSL-style members (x/y/z/w, r/g/b/a) and
// arithmetic. Constructors are implicit on purpose to keep ported HLSL code
// unmodified.
// ---------------------------------------------------------------------------

template <typename T, int N>
struct TVec;

template <typename T>
struct TVec<T, 2> {
  // Only scalar member aliases (x/y vs r/g) live in the anonymous union;
  // nested TVec sub-members (.xy, .xyz, ...) are NOT provided: GCC 12/13 on
  // x86-64 miscompile code returning these types by value when the union
  // contains non-trivial nested TVec members. The transpiler expands all
  // multi-component swizzles into indexed component reads, so the sub-vector
  // aliases are never needed in generated code. Single-component .r/.g/.b/.a
  // access is emitted as-is by the transpiler and relies on these aliases.
  union {
    struct {
      T x, y;
    };
    struct {
      T r, g;
    };
  };
  SPARKIUM_KERNEL TVec() : x(0), y(0) {}
  SPARKIUM_KERNEL TVec(T v) : x(v), y(v) {}
  template <typename A, typename B>
  SPARKIUM_KERNEL TVec(A a, B b) : x(T(a)), y(T(b)) {}
  template <typename U>
  SPARKIUM_KERNEL explicit TVec(const TVec<U, 2> &o) : x(T(o.x)), y(T(o.y)) {}
  SPARKIUM_KERNEL T &operator[](int i) { return (&x)[i]; }
  SPARKIUM_KERNEL T operator[](int i) const { return (&x)[i]; }
};

template <typename T>
struct TVec<T, 3> {
  union {
    struct {
      T x, y, z;
    };
    struct {
      T r, g, b;
    };
  };
  SPARKIUM_KERNEL TVec() : x(0), y(0), z(0) {}
  SPARKIUM_KERNEL TVec(T v) : x(v), y(v), z(v) {}
  SPARKIUM_KERNEL TVec(const TVec<T, 2> &a, T b) : x(a.x), y(a.y), z(b) {}
  SPARKIUM_KERNEL TVec(T a, const TVec<T, 2> &b) : x(a), y(b.x), z(b.y) {}
  template <typename A, typename B, typename C>
  SPARKIUM_KERNEL TVec(A a, B b, C c) : x(T(a)), y(T(b)), z(T(c)) {}
  template <typename U>
  SPARKIUM_KERNEL explicit TVec(const TVec<U, 3> &o) : x(T(o.x)), y(T(o.y)), z(T(o.z)) {}
  SPARKIUM_KERNEL T &operator[](int i) { return (&x)[i]; }
  SPARKIUM_KERNEL T operator[](int i) const { return (&x)[i]; }
};

template <typename T>
struct TVec<T, 4> {
  union {
    struct {
      T x, y, z, w;
    };
    struct {
      T r, g, b, a;
    };
  };
  SPARKIUM_KERNEL TVec() : x(0), y(0), z(0), w(0) {}
  SPARKIUM_KERNEL TVec(T v) : x(v), y(v), z(v), w(v) {}
  SPARKIUM_KERNEL TVec(const TVec<T, 3> &a, T b) : x(a.x), y(a.y), z(a.z), w(b) {}
  // HLSL permits assigning a smaller vector to a larger one (zero padding).
  SPARKIUM_KERNEL TVec(const TVec<T, 3> &a) : x(a.x), y(a.y), z(a.z), w(T(0)) {}
  SPARKIUM_KERNEL TVec(T a, const TVec<T, 3> &b) : x(a), y(b.x), z(b.y), w(b.z) {}
  SPARKIUM_KERNEL TVec(const TVec<T, 2> &a, const TVec<T, 2> &b) : x(a.x), y(a.y), z(b.x), w(b.y) {}
  SPARKIUM_KERNEL TVec(T a, const TVec<T, 2> &b, T c) : x(a), y(b.x), z(b.y), w(c) {}
  SPARKIUM_KERNEL TVec(T a, T b, const TVec<T, 2> &c) : x(a), y(b), z(c.x), w(c.y) {}
  template <typename A, typename B, typename C, typename D>
  SPARKIUM_KERNEL TVec(A a, B b, C c, D d) : x(T(a)), y(T(b)), z(T(c)), w(T(d)) {}
  template <typename U>
  SPARKIUM_KERNEL explicit TVec(const TVec<U, 4> &o) : x(T(o.x)), y(T(o.y)), z(T(o.z)), w(T(o.w)) {}
  SPARKIUM_KERNEL T &operator[](int i) { return (&x)[i]; }
  SPARKIUM_KERNEL T operator[](int i) const { return (&x)[i]; }
};

typedef TVec<float, 2> float2;
typedef TVec<float, 3> float3;
typedef TVec<float, 4> float4;
typedef TVec<int32_t, 2> int2;
typedef TVec<int32_t, 3> int3;
typedef TVec<int32_t, 4> int4;
typedef TVec<uint, 2> uint2;
typedef TVec<uint, 3> uint3;
typedef TVec<uint, 4> uint4;

// ---------------------------------------------------------------------------
// Vector arithmetic
// ---------------------------------------------------------------------------

#define SPARKIUM_VEC_OPS(N)                                                                             \
  template <typename T>                                                                                 \
  SPARKIUM_KERNEL TVec<T, N> operator+(const TVec<T, N> &a, const TVec<T, N> &b) {                      \
    TVec<T, N> r;                                                                                       \
    for (int i = 0; i < N; ++i)                                                                         \
      r[i] = a[i] + b[i];                                                                               \
    return r;                                                                                           \
  }                                                                                                     \
  template <typename T>                                                                                 \
  SPARKIUM_KERNEL TVec<T, N> operator-(const TVec<T, N> &a, const TVec<T, N> &b) {                      \
    TVec<T, N> r;                                                                                       \
    for (int i = 0; i < N; ++i)                                                                         \
      r[i] = a[i] - b[i];                                                                               \
    return r;                                                                                           \
  }                                                                                                     \
  template <typename T>                                                                                 \
  SPARKIUM_KERNEL TVec<T, N> operator*(const TVec<T, N> &a, const TVec<T, N> &b) {                      \
    TVec<T, N> r;                                                                                       \
    for (int i = 0; i < N; ++i)                                                                         \
      r[i] = a[i] * b[i];                                                                               \
    return r;                                                                                           \
  }                                                                                                     \
  template <typename T>                                                                                 \
  SPARKIUM_KERNEL TVec<T, N> operator/(const TVec<T, N> &a, const TVec<T, N> &b) {                      \
    TVec<T, N> r;                                                                                       \
    for (int i = 0; i < N; ++i)                                                                         \
      r[i] = a[i] / b[i];                                                                               \
    return r;                                                                                           \
  }                                                                                                     \
  template <typename T>                                                                                 \
  SPARKIUM_KERNEL TVec<T, N> operator-(const TVec<T, N> &a) {                                             \
    TVec<T, N> r;                                                                                       \
    for (int i = 0; i < N; ++i)                                                                         \
      r[i] = -a[i];                                                                                     \
    return r;                                                                                           \
  }                                                                                                     \
  template <typename T, typename S>                                                                     \
  SPARKIUM_KERNEL TVec<T, N> operator*(const TVec<T, N> &a, S s) {                                        \
    TVec<T, N> r;                                                                                       \
    for (int i = 0; i < N; ++i)                                                                         \
      r[i] = a[i] * T(s);                                                                               \
    return r;                                                                                           \
  }                                                                                                     \
  template <typename T, typename S>                                                                     \
  SPARKIUM_KERNEL TVec<T, N> operator*(S s, const TVec<T, N> &a) {                                        \
    return a * s;                                                                                       \
  }                                                                                                     \
  template <typename T, typename S>                                                                     \
  SPARKIUM_KERNEL TVec<T, N> operator/(const TVec<T, N> &a, S s) {                                        \
    TVec<T, N> r;                                                                                       \
    for (int i = 0; i < N; ++i)                                                                         \
      r[i] = a[i] / T(s);                                                                               \
    return r;                                                                                           \
  }                                                                                                     \
  template <typename T, typename S>                                                                     \
  SPARKIUM_KERNEL TVec<T, N> operator+(const TVec<T, N> &a, S s) {                                        \
    TVec<T, N> r;                                                                                       \
    for (int i = 0; i < N; ++i)                                                                         \
      r[i] = a[i] + T(s);                                                                               \
    return r;                                                                                           \
  }                                                                                                     \
  template <typename T, typename S>                                                                     \
  SPARKIUM_KERNEL TVec<T, N> operator+(S s, const TVec<T, N> &a) {                                        \
    return a + s;                                                                                       \
  }                                                                                                     \
  template <typename T, typename S>                                                                     \
  SPARKIUM_KERNEL TVec<T, N> operator-(const TVec<T, N> &a, S s) {                                        \
    TVec<T, N> r;                                                                                       \
    for (int i = 0; i < N; ++i)                                                                         \
      r[i] = a[i] - T(s);                                                                               \
    return r;                                                                                           \
  }                                                                                                     \
  template <typename T, typename S>                                                                     \
  SPARKIUM_KERNEL TVec<T, N> operator-(S s, const TVec<T, N> &a) {                                        \
    TVec<T, N> r;                                                                                       \
    for (int i = 0; i < N; ++i)                                                                         \
      r[i] = T(s) - a[i];                                                                               \
    return r;                                                                                           \
  }                                                                                                     \
  template <typename T>                                                                                 \
  SPARKIUM_KERNEL TVec<T, N> &operator+=(TVec<T, N> &a, const TVec<T, N> &b) {                            \
    for (int i = 0; i < N; ++i)                                                                         \
      a[i] += b[i];                                                                                     \
    return a;                                                                                           \
  }                                                                                                     \
  template <typename T>                                                                                 \
  SPARKIUM_KERNEL TVec<T, N> &operator-=(TVec<T, N> &a, const TVec<T, N> &b) {                            \
    for (int i = 0; i < N; ++i)                                                                         \
      a[i] -= b[i];                                                                                     \
    return a;                                                                                           \
  }                                                                                                     \
  template <typename T>                                                                                 \
  SPARKIUM_KERNEL TVec<T, N> &operator*=(TVec<T, N> &a, const TVec<T, N> &b) {                            \
    for (int i = 0; i < N; ++i)                                                                         \
      a[i] *= b[i];                                                                                     \
    return a;                                                                                           \
  }                                                                                                     \
  template <typename T, typename S>                                                                     \
  SPARKIUM_KERNEL TVec<T, N> &operator*=(TVec<T, N> &a, S s) {                                            \
    for (int i = 0; i < N; ++i)                                                                         \
      a[i] *= T(s);                                                                                     \
    return a;                                                                                           \
  }                                                                                                     \
  template <typename T, typename S>                                                                     \
  SPARKIUM_KERNEL TVec<T, N> &operator/=(TVec<T, N> &a, S s) {                                            \
    for (int i = 0; i < N; ++i)                                                                         \
      a[i] /= T(s);                                                                                     \
    return a;                                                                                           \
  }                                                                                                     \
  template <typename T>                                                                                 \
  SPARKIUM_KERNEL bool operator==(const TVec<T, N> &a, const TVec<T, N> &b) {                             \
    for (int i = 0; i < N; ++i)                                                                         \
      if (a[i] != b[i])                                                                                 \
        return false;                                                                                   \
    return true;                                                                                        \
  }                                                                                                     \
  template <typename T>                                                                                 \
  SPARKIUM_KERNEL bool operator!=(const TVec<T, N> &a, const TVec<T, N> &b) {                             \
    return !(a == b);                                                                                   \
  }

SPARKIUM_VEC_OPS(2)
SPARKIUM_VEC_OPS(3)
SPARKIUM_VEC_OPS(4)

// ---------------------------------------------------------------------------
// Intrinsics (v-prefixed; the transpiler rewrites HLSL names onto these).
// ---------------------------------------------------------------------------

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vmin(const TVec<T, N> &a, const TVec<T, N> &b) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = a[i] < b[i] ? a[i] : b[i];
  return r;
}
template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vmax(const TVec<T, N> &a, const TVec<T, N> &b) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = a[i] > b[i] ? a[i] : b[i];
  return r;
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vabs(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = a[i] < T(0) ? -a[i] : a[i];
  return r;
}

template <typename T, int N>
SPARKIUM_KERNEL T dot(const TVec<T, N> &a, const TVec<T, N> &b) {
  T r(0);
  for (int i = 0; i < N; ++i)
    r += a[i] * b[i];
  return r;
}

SPARKIUM_KERNEL inline float3 cross(const float3 &a, const float3 &b) {
  return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

template <typename T, int N>
SPARKIUM_KERNEL T vlength(const TVec<T, N> &a) {
  return std::sqrt(dot(a, a));
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vnormalize(const TVec<T, N> &a) {
  return a / vlength(a);
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vsaturate(const TVec<T, N> &a) {
  return vmax(vmin(a, TVec<T, N>(T(1))), TVec<T, N>(T(0)));
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vclamp(const TVec<T, N> &a, const TVec<T, N> &lo, const TVec<T, N> &hi) {
  return vmax(vmin(a, hi), lo);
}
template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vlerp(const TVec<T, N> &a, const TVec<T, N> &b, const TVec<T, N> &t) {
  return a + (b - a) * t;
}

// Element-wise comparisons produce boolean masks (int vectors), HLSL style.
#define SPARKIUM_VEC_CMP(N)                                                                          \
  template <typename T>                                                                              \
  SPARKIUM_KERNEL TVec<int, N> operator>(const TVec<T, N> &a, const TVec<T, N> &b) {                 \
    TVec<int, N> r;                                                                                  \
    for (int i = 0; i < N; ++i)                                                                      \
      r[i] = a[i] > b[i];                                                                            \
    return r;                                                                                        \
  }                                                                                                  \
  template <typename T>                                                                              \
  SPARKIUM_KERNEL TVec<int, N> operator<(const TVec<T, N> &a, const TVec<T, N> &b) {                 \
    TVec<int, N> r;                                                                                  \
    for (int i = 0; i < N; ++i)                                                                      \
      r[i] = a[i] < b[i];                                                                            \
    return r;                                                                                        \
  }                                                                                                  \
  template <typename T>                                                                              \
  SPARKIUM_KERNEL TVec<int, N> operator>=(const TVec<T, N> &a, const TVec<T, N> &b) {                \
    TVec<int, N> r;                                                                                  \
    for (int i = 0; i < N; ++i)                                                                      \
      r[i] = a[i] >= b[i];                                                                           \
    return r;                                                                                        \
  }                                                                                                  \
  template <typename T>                                                                              \
  SPARKIUM_KERNEL TVec<int, N> operator<=(const TVec<T, N> &a, const TVec<T, N> &b) {                \
    TVec<int, N> r;                                                                                  \
    for (int i = 0; i < N; ++i)                                                                      \
      r[i] = a[i] <= b[i];                                                                           \
    return r;                                                                                        \
  }                                                                                                  \
  template <typename T>                                                                              \
  SPARKIUM_KERNEL TVec<int, N> operator==(const TVec<T, N> &a, const TVec<T, N> &b) {                \
    TVec<int, N> r;                                                                                  \
    for (int i = 0; i < N; ++i)                                                                      \
      r[i] = a[i] == b[i];                                                                           \
    return r;                                                                                        \
  }                                                                                                  \
  template <typename T>                                                                              \
  SPARKIUM_KERNEL TVec<int, N> operator!=(const TVec<T, N> &a, const TVec<T, N> &b) {                \
    TVec<int, N> r;                                                                                  \
    for (int i = 0; i < N; ++i)                                                                      \
      r[i] = a[i] != b[i];                                                                           \
    return r;                                                                                        \
  }
SPARKIUM_VEC_CMP(2)
SPARKIUM_VEC_CMP(3)
SPARKIUM_VEC_CMP(4)

template <typename T, int N>
SPARKIUM_KERNEL bool vall(const TVec<T, N> &a) {
  for (int i = 0; i < N; ++i)
    if (!a[i])
      return false;
  return true;
}

template <typename T, int N>
SPARKIUM_KERNEL bool vany(const TVec<T, N> &a) {
  for (int i = 0; i < N; ++i)
    if (a[i])
      return true;
  return false;
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vfrac(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = a[i] - std::floor(a[i]);
  return r;
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vfloor(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::floor(a[i]);
  return r;
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vpow(const TVec<T, N> &a, const TVec<T, N> &b) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::pow(a[i], b[i]);
  return r;
}

template <typename T, int N, typename S>
SPARKIUM_KERNEL TVec<T, N> vpows(const TVec<T, N> &a, S b) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::pow(a[i], T(b));
  return r;
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vexp(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::exp(a[i]);
  return r;
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vlog(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::log(a[i]);
  return r;
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vsin(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::sin(a[i]);
  return r;
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vcos(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::cos(a[i]);
  return r;
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vtan(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::tan(a[i]);
  return r;
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vsqrt(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::sqrt(a[i]);
  return r;
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vatan2(const TVec<T, N> &a, const TVec<T, N> &b) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::atan2(a[i], b[i]);
  return r;
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vatan(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::atan(a[i]);
  return r;
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vacos(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::acos(a[i]);
  return r;
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vasin(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::asin(a[i]);
  return r;
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vfmod(const TVec<T, N> &a, const TVec<T, N> &b) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::fmod(a[i], b[i]);
  return r;
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vstep(const TVec<T, N> &edge, const TVec<T, N> &x) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = x[i] < edge[i] ? T(0) : T(1);
  return r;
}
template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vceil(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::ceil(a[i]);
  return r;
}
template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vtrunc(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::trunc(a[i]);
  return r;
}
template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vround(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::round(a[i]);
  return r;
}
template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vexp2(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::exp2(a[i]);
  return r;
}
template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vlog2(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::log2(a[i]);
  return r;
}
template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vrsqrt(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = T(1) / std::sqrt(a[i]);
  return r;
}
template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> visnan(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::isnan(a[i]) ? T(1) : T(0);
  return r;
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vsign(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = a[i] > T(0) ? T(1) : (a[i] < T(0) ? T(-1) : T(0));
  return r;
}

SPARKIUM_KERNEL inline float3 vreflect(const float3 &i, const float3 &n) {
  return i - n * (2.0f * dot(i, n));
}

template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> visfinite(const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = std::isfinite(a[i]) ? T(1) : T(0);
  return r;
}


// Scalar overloads so transpiled HLSL works unchanged for float arguments.
SPARKIUM_KERNEL inline float vsin(float a) { return std::sin(a); }
SPARKIUM_KERNEL inline float vcos(float a) { return std::cos(a); }
SPARKIUM_KERNEL inline float vtan(float a) { return std::tan(a); }
SPARKIUM_KERNEL inline float vsqrt(float a) { return std::sqrt(a); }
SPARKIUM_KERNEL inline float vabs(float a) { return std::fabs(a); }
SPARKIUM_KERNEL inline float vexp(float a) { return std::exp(a); }
SPARKIUM_KERNEL inline float vlog(float a) { return std::log(a); }
SPARKIUM_KERNEL inline float vpow(float a, float b) { return std::pow(a, b); }
SPARKIUM_KERNEL inline float vpow(float a, int b) { return std::pow(a, float(b)); }
SPARKIUM_KERNEL inline float vpow(float a, double b) { return std::pow(a, float(b)); }
SPARKIUM_KERNEL inline float vfloor(float a) { return std::floor(a); }
SPARKIUM_KERNEL inline float vceil(float a) { return std::ceil(a); }
SPARKIUM_KERNEL inline float vround(float a) { return std::round(a); }
SPARKIUM_KERNEL inline float vtrunc(float a) { return std::trunc(a); }
SPARKIUM_KERNEL inline float vfrac(float a) { return a - std::floor(a); }
SPARKIUM_KERNEL inline float vsaturate(float a) { return a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a); }
SPARKIUM_KERNEL inline float vatan2(float a, float b) { return std::atan2(a, b); }
SPARKIUM_KERNEL inline float vatan(float a) { return std::atan(a); }
SPARKIUM_KERNEL inline float vacos(float a) { return std::acos(a); }
SPARKIUM_KERNEL inline float vasin(float a) { return std::asin(a); }
SPARKIUM_KERNEL inline float vfmod(float a, float b) { return std::fmod(a, b); }
SPARKIUM_KERNEL inline float vstep(float edge, float x) { return x < edge ? 0.0f : 1.0f; }
SPARKIUM_KERNEL inline float vlerp(float a, float b, float t) { return a + (b - a) * t; }
SPARKIUM_KERNEL inline float vmin(float a, float b) { return a < b ? a : b; }
SPARKIUM_KERNEL inline float vmax(float a, float b) { return a > b ? a : b; }
SPARKIUM_KERNEL inline float vclamp(float x, float lo, float hi) { return vmin(vmax(x, lo), hi); }
SPARKIUM_KERNEL inline float vrsqrt(float a) { return 1.0f / std::sqrt(a); }
SPARKIUM_KERNEL inline float vexp2(float a) { return std::exp2(a); }
SPARKIUM_KERNEL inline float vlog2(float a) { return std::log2(a); }
SPARKIUM_KERNEL inline float vsign(float a) { return a > 0.0f ? 1.0f : (a < 0.0f ? -1.0f : 0.0f); }
SPARKIUM_KERNEL inline float vmad(float a, float b, float c) { return a * b + c; }
SPARKIUM_KERNEL inline float vrcp(float a) { return 1.0f / a; }
SPARKIUM_KERNEL inline double vsin(double a) { return std::sin(a); }
SPARKIUM_KERNEL inline double vcos(double a) { return std::cos(a); }
SPARKIUM_KERNEL inline double vsqrt(double a) { return std::sqrt(a); }

SPARKIUM_KERNEL inline float vpows(float a, float b) { return std::pow(a, b); }
SPARKIUM_KERNEL inline float vpows(float a, int b) { return std::pow(a, float(b)); }
SPARKIUM_KERNEL inline float vpows(float a, double b) { return std::pow(a, float(b)); }
SPARKIUM_KERNEL inline bool visfinite(float a) { return std::isfinite(a); }
SPARKIUM_KERNEL inline bool visnan(float a) { return std::isnan(a); }

// Device-safe scalar min/max. std::min/std::max are constexpr __host__
// functions under nvcc and cannot be called from __device__ code, so the
// vector helpers below go through these wrappers instead.
template <typename T>
SPARKIUM_KERNEL inline T ScalarMin(T a, T b) {
  return a < b ? a : b;
}
template <typename T>
SPARKIUM_KERNEL inline T ScalarMax(T a, T b) {
  return a < b ? b : a;
}

template <typename T, int N, typename S>
SPARKIUM_KERNEL TVec<T, N> vclamp(const TVec<T, N> &a, S lo, S hi) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = ScalarMin(ScalarMax(a[i], T(lo)), T(hi));
  return r;
}
template <typename T, int N, typename S>
SPARKIUM_KERNEL TVec<T, N> vmin(const TVec<T, N> &a, S b) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = ScalarMin(a[i], T(b));
  return r;
}
template <typename T, int N, typename S>
SPARKIUM_KERNEL TVec<T, N> vmax(const TVec<T, N> &a, S b) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = ScalarMax(a[i], T(b));
  return r;
}
template <typename S, typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vmin(S a, const TVec<T, N> &b) {
  return vmin(b, a);
}
template <typename S, typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vmax(S a, const TVec<T, N> &b) {
  return vmax(b, a);
}

template <typename T, int N, typename S>
SPARKIUM_KERNEL TVec<T, N> vlerp(const TVec<T, N> &a, const TVec<T, N> &b, S t) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = a[i] + (b[i] - a[i]) * T(t);
  return r;
}
template <typename S, typename T, int N>
SPARKIUM_KERNEL TVec<T, N> vlerp(S a, S b, const TVec<T, N> &t) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = T(a) + (T(b) - T(a)) * t[i];
  return r;
}
// Cross-type vector assignment (HLSL converts implicitly).
template <typename T, typename U>
SPARKIUM_KERNEL void AssignVec(TVec<T, 3> &dst, const TVec<U, 3> &src) {
  dst.x = T(src.x);
  dst.y = T(src.y);
  dst.z = T(src.z);
}
#define NonUniformResourceIndex(x) (x)

SPARKIUM_KERNEL inline float vclampf(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// ---------------------------------------------------------------------------
// Bit casts
// ---------------------------------------------------------------------------

SPARKIUM_KERNEL inline float asfloat(uint u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
SPARKIUM_KERNEL inline float asfloat(int32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
SPARKIUM_KERNEL inline uint asuint(float f) {
  uint u;
  std::memcpy(&u, &f, 4);
  return u;
}
SPARKIUM_KERNEL inline int32_t asint(float f) {
  int32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
SPARKIUM_KERNEL inline float2 asfloat(const uint2 &v) {
  return float2(asfloat(v.x), asfloat(v.y));
}
SPARKIUM_KERNEL inline uint2 asuint(const float2 &v) {
  return uint2(asuint(v.x), asuint(v.y));
}
SPARKIUM_KERNEL inline float3 asfloat(const uint3 &v) {
  return float3(asfloat(v.x), asfloat(v.y), asfloat(v.z));
}
SPARKIUM_KERNEL inline uint3 asuint(const float3 &v) {
  return uint3(asuint(v.x), asuint(v.y), asuint(v.z));
}
SPARKIUM_KERNEL inline float4 asfloat(const uint4 &v) {
  return float4(asfloat(v.x), asfloat(v.y), asfloat(v.z), asfloat(v.w));
}
SPARKIUM_KERNEL inline uint4 asuint(const float4 &v) {
  return uint4(asuint(v.x), asuint(v.y), asuint(v.z), asuint(v.w));
}
// Identity casts used by HLSL buffer helpers.
SPARKIUM_KERNEL inline uint2 asuint(const uint2 &v) { return v; }
SPARKIUM_KERNEL inline uint3 asuint(const uint3 &v) { return v; }
SPARKIUM_KERNEL inline uint4 asuint(const uint4 &v) { return v; }
SPARKIUM_KERNEL inline uint2 asuint(const int2 &v) { return uint2(uint(v.x), uint(v.y)); }
SPARKIUM_KERNEL inline uint3 asuint(const int3 &v) { return uint3(uint(v.x), uint(v.y), uint(v.z)); }
SPARKIUM_KERNEL inline uint4 asuint(const int4 &v) { return uint4(uint(v.x), uint(v.y), uint(v.z), uint(v.w)); }
SPARKIUM_KERNEL inline float2 asfloat(const int2 &v) {
  return float2(asfloat(v.x), asfloat(v.y));
}
SPARKIUM_KERNEL inline float3 asfloat(const int3 &v) {
  return float3(asfloat(v.x), asfloat(v.y), asfloat(v.z));
}
SPARKIUM_KERNEL inline float4 asfloat(const int4 &v) {
  return float4(asfloat(v.x), asfloat(v.y), asfloat(v.z), asfloat(v.w));
}
SPARKIUM_KERNEL inline float2 asfloat(const float2 &v) { return v; }
SPARKIUM_KERNEL inline float3 asfloat(const float3 &v) { return v; }

// Component-wise bit cast used for the HLSL idiom asfloat(floatN(uint...)).
// The transpiler expands multi-component swizzle reads into indexed component
// reads, which turns asfloat(v.xyz) into asfloat(floatN(v[0], v[1], ...)).
// In HLSL the outer asfloat still bit-casts every component, so this helper
// performs that cast; a plain floatN(...) constructor would instead convert
// the integer values numerically.
SPARKIUM_KERNEL inline float SparkiumAsFloat(uint a) { return asfloat(a); }
SPARKIUM_KERNEL inline float2 SparkiumAsFloat(uint a, uint b) { return float2(asfloat(a), asfloat(b)); }
SPARKIUM_KERNEL inline float3 SparkiumAsFloat(uint a, uint b, uint c) {
  return float3(asfloat(a), asfloat(b), asfloat(c));
}
SPARKIUM_KERNEL inline float4 SparkiumAsFloat(uint a, uint b, uint c, uint d) {
  return float4(asfloat(a), asfloat(b), asfloat(c), asfloat(d));
}
SPARKIUM_KERNEL inline float4 asfloat(const float4 &v) { return v; }

SPARKIUM_KERNEL inline float asfloat(bool b) {
  return b ? 1.0f : 0.0f;
}

// ---------------------------------------------------------------------------
// HLSL max/min survive inside macro expansions as unqualified calls.
#ifdef __CUDACC__
using ::max;
using ::min;
#else
using std::max;
using std::min;
#endif

// Scalar / vector division (HLSL element-wise reciprocal).
template <typename S, typename T, int N>
SPARKIUM_KERNEL TVec<T, N> operator/(S s, const TVec<T, N> &a) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = T(s) / a[i];
  return r;
}

// Built-in HLSL ray-tracing types.
struct RayDesc {
  float3 Origin;
  float3 Direction;
  float TMin{0.0f};
  float TMax{0.0f};
};

// Cross-type scalar/vector conversion helpers used by transpiled HLSL
// compound swizzle assignments.
template <typename T, int N>
SPARKIUM_KERNEL TVec<T, N> VecFromScalar(float v) {
  TVec<T, N> r;
  for (int i = 0; i < N; ++i)
    r[i] = T(v);
  return r;
}

// Multi-component swizzle assignment helper (type-converting, HLSL
// semantics): dst[i_k] = T(src[k]).
template <typename T, int N, typename U, int M, typename... Idx>
SPARKIUM_KERNEL void SparkiumSwizzleAssignImpl(TVec<T, N> &dst, const TVec<U, M> &src, int k, int idx, Idx... rest) {
  dst[idx] = T(src[k]);
  if constexpr (sizeof...(rest) > 0)
    SparkiumSwizzleAssignImpl(dst, src, k + 1, rest...);
}
template <typename T, int N, typename U, int M, typename... Idx>
SPARKIUM_KERNEL void SparkiumSwizzleAssign(TVec<T, N> &dst, const TVec<U, M> &src, int count, Idx... idx) {
  static_assert(sizeof...(idx) > 0, "swizzle assign needs indices");
  SparkiumSwizzleAssignImpl(dst, src, 0, idx...);
}

// Column-major matrices mirroring HLSL floatMxN (M rows, N columns).
// ---------------------------------------------------------------------------

template <int M, int N>
struct TMat {
  TVec<float, M> columns[N];
  SPARKIUM_KERNEL TMat() = default;
  // HLSL-style constructor from column vectors.
  template <typename... Cols, typename = std::enable_if_t<sizeof...(Cols) == N>>
  SPARKIUM_KERNEL TMat(Cols... cols) : columns{TVec<float, M>(cols)...} {}
  SPARKIUM_KERNEL TVec<float, M> &operator[](int i) { return columns[i]; }
  SPARKIUM_KERNEL const TVec<float, M> &operator[](int i) const { return columns[i]; }
};

typedef TMat<3, 3> float3x3;
typedef TMat<4, 4> float4x4;
typedef TMat<3, 4> float3x4;
typedef TMat<4, 3> float4x3;
typedef TMat<2, 2> float2x2;
typedef TMat<4, 2> float4x2;
typedef TMat<2, 4> float2x4;
typedef TMat<2, 3> float2x3;
typedef TMat<3, 2> float3x2;

template <int M, int N>
SPARKIUM_KERNEL TVec<float, M> mul(const TMat<M, N> &a, const TVec<float, N> &v) {
  TVec<float, M> r;
  for (int m = 0; m < M; ++m) {
    float s = 0.0f;
    for (int n = 0; n < N; ++n)
      s += a[n][m] * v[n];
    r[m] = s;
  }
  return r;
}

template <int M, int N>
SPARKIUM_KERNEL TVec<float, M> mul(const TVec<float, N> &v, const TMat<M, N> &a) {
  TVec<float, M> r;
  for (int m = 0; m < M; ++m) {
    float s = 0.0f;
    for (int n = 0; n < N; ++n)
      s += v[n] * a[n][m];
    r[m] = s;
  }
  return r;
}

template <int M, int N>
SPARKIUM_KERNEL TMat<M, N> matmul_scalar(const TMat<M, N> &a, float s) {
  TMat<M, N> r;
  for (int n = 0; n < N; ++n)
    for (int m = 0; m < M; ++m)
      r[n][m] = a[n][m] * s;
  return r;
}

template <int M, int N>
SPARKIUM_KERNEL TMat<N, M> transpose(const TMat<M, N> &a) {
  TMat<N, M> r;
  // NOTE: routing each component through a named scalar local (instead of
  // writing r.columns[m][n] = a.columns[n][m] directly) avoids a GCC x86-64
  // codegen bug (observed with GCC 13.3, -O1/-O2) in which stores to the
  // result overwrite the function's own stack-protector canary slot and
  // bytes beyond the returned object, corrupting the caller's stack. The
  // scalars live in registers, so the generated code is identical in
  // performance.
  for (int n = 0; n < N; ++n)
    for (int m = 0; m < M; ++m) {
      const float component = a.columns[n][m];
      r.columns[m][n] = component;
    }
  return r;
}

// ---------------------------------------------------------------------------
// Byte-addressable buffer mirroring HLSL ByteAddressBuffer Load/Store.
// ---------------------------------------------------------------------------

// NOTE: GCC 12/13 on x86-64 miscompiles (wrong-value, not just stack
// protection noise) functions that return small vectors constructed as
// `uintN(Load(a), Load(b), ...)` directly from Load() call results: the
// compiler reuses the base-offset register for a component load after
// clobbering it. Routing every component through a named scalar local
// produces correct code on all tested compilers and is free (the scalars
// stay in registers). Keep this pattern in every LoadN below.
struct ByteAddressBuffer {
  const uint8_t *data{nullptr};
  size_t size{0};

  SPARKIUM_KERNEL uint Load(uint offset) const {
    uint v = 0;
    if (offset + 4 <= size)
      std::memcpy(&v, data + offset, 4);
    return v;
  }
  SPARKIUM_KERNEL uint2 Load2(uint offset) const {
    const uint b0 = Load(offset);
    const uint b1 = Load(offset + 4);
    return uint2(b0, b1);
  }
  SPARKIUM_KERNEL uint3 Load3(uint offset) const {
    const uint b0 = Load(offset);
    const uint b1 = Load(offset + 4);
    const uint b2 = Load(offset + 8);
    return uint3(b0, b1, b2);
  }
  SPARKIUM_KERNEL uint4 Load4(uint offset) const {
    const uint b0 = Load(offset);
    const uint b1 = Load(offset + 4);
    const uint b2 = Load(offset + 8);
    const uint b3 = Load(offset + 12);
    return uint4(b0, b1, b2, b3);
  }
  template <typename T>
  SPARKIUM_KERNEL T Load(uint offset) const {
    T v;
    std::memset(&v, 0, sizeof(T));
    if (offset + sizeof(T) <= size)
      std::memcpy(&v, data + offset, sizeof(T));
    return v;
  }
  template <typename T>
  SPARKIUM_KERNEL T Load() const {
    return Load<T>(0);
  }
};

struct RWByteAddressBuffer {
  uint8_t *data{nullptr};
  size_t size{0};

  SPARKIUM_KERNEL operator ByteAddressBuffer() const {
    return ByteAddressBuffer{data, size};
  }
  SPARKIUM_KERNEL uint Load(uint offset) const {
    uint v = 0;
    if (offset + 4 <= size)
      std::memcpy(&v, data + offset, 4);
    return v;
  }
  SPARKIUM_KERNEL uint2 Load2(uint offset) const {
    const uint b0 = Load(offset);
    const uint b1 = Load(offset + 4);
    return uint2(b0, b1);
  }
  SPARKIUM_KERNEL uint3 Load3(uint offset) const {
    const uint b0 = Load(offset);
    const uint b1 = Load(offset + 4);
    const uint b2 = Load(offset + 8);
    return uint3(b0, b1, b2);
  }
  SPARKIUM_KERNEL uint4 Load4(uint offset) const {
    const uint b0 = Load(offset);
    const uint b1 = Load(offset + 4);
    const uint b2 = Load(offset + 8);
    const uint b3 = Load(offset + 12);
    return uint4(b0, b1, b2, b3);
  }
  template <typename T>
  SPARKIUM_KERNEL T Load(uint offset) const {
    T v;
    std::memset(&v, 0, sizeof(T));
    if (offset + sizeof(T) <= size)
      std::memcpy(&v, data + offset, sizeof(T));
    return v;
  }
  SPARKIUM_KERNEL void Store(uint offset, uint value) const {
    if (offset + 4 <= size)
      std::memcpy(data + offset, &value, 4);
  }
  SPARKIUM_KERNEL void Store2(uint offset, uint2 value) const {
    Store(offset, value.x);
    Store(offset + 4, value.y);
  }
  SPARKIUM_KERNEL void Store3(uint offset, uint3 value) const {
    Store(offset, value.x);
    Store(offset + 4, value.y);
    Store(offset + 8, value.z);
  }
  SPARKIUM_KERNEL void Store4(uint offset, uint4 value) const {
    Store(offset, value.x);
    Store(offset + 4, value.y);
    Store(offset + 8, value.z);
    Store(offset + 12, value.w);
  }
  template <typename T>
  SPARKIUM_KERNEL void Store(uint offset, const T &value) const {
    if (offset + sizeof(T) <= size)
      std::memcpy(data + offset, &value, sizeof(T));
  }
};

// Byte-addressed views over typed storage with identical load semantics.
struct FloatBuffer {
  const float *data{nullptr};
  size_t count{0};
  SPARKIUM_KERNEL float Load(uint index) const {
    return index < count ? data[index] : 0.0f;
  }
  template <typename T>
  SPARKIUM_KERNEL T Load(uint index) const {
    T v;
    std::memset(&v, 0, sizeof(T));
    const uint bytes = index * 4;
    if (bytes + sizeof(T) <= count * 4)
      std::memcpy(&v, reinterpret_cast<const uint8_t *>(data) + bytes, sizeof(T));
    return v;
  }
};

struct RWFloatBuffer {
  float *data{nullptr};
  size_t count{0};
  SPARKIUM_KERNEL float Load(uint index) const {
    return index < count ? data[index] : 0.0f;
  }
  SPARKIUM_KERNEL void Store(uint index, float value) const {
    if (index < count)
      data[index] = value;
  }
};

// ---------------------------------------------------------------------------
// Texture2D with a linear-repeat sampler (matches samplers[0] of the GPU
// pipelines).
// ---------------------------------------------------------------------------

enum TextureFormat : uint { TEXTURE_FORMAT_SDR = 0, TEXTURE_FORMAT_HDR = 1 };

struct Texture2D {
  const void *data{nullptr};
  uint width{0};
  uint height{0};
  TextureFormat format{TEXTURE_FORMAT_SDR};

  SPARKIUM_KERNEL float4 Texel(uint x, uint y) const {
    if (format == TEXTURE_FORMAT_SDR) {
      const uint8_t *p = static_cast<const uint8_t *>(data) + (size_t(y) * width + x) * 4;
      return float4(p[0] / 255.0f, p[1] / 255.0f, p[2] / 255.0f, p[3] / 255.0f);
    }
    const float *p = static_cast<const float *>(data) + (size_t(y) * width + x) * 4;
    return float4(p[0], p[1], p[2], p[3]);
  }

  SPARKIUM_KERNEL float4 SampleBilinear(float2 uv) const {
    if (!data || width == 0 || height == 0)
      return float4(1.0f, 0.0f, 1.0f, 1.0f);
    const float fx = uv.x * float(width) - 0.5f;
    const float fy = uv.y * float(height) - 0.5f;
    const int ix = int(std::floor(fx));
    const int iy = int(std::floor(fy));
    const float tx = fx - float(ix);
    const float ty = fy - float(iy);
    auto wrap = [](int v, int n) {
      v %= n;
      return v < 0 ? v + n : v;
    };
    const int x0 = wrap(ix, int(width)), x1 = wrap(ix + 1, int(width));
    const int y0 = wrap(iy, int(height)), y1 = wrap(iy + 1, int(height));
    const float4 c00 = Texel(uint(x0), uint(y0));
    const float4 c10 = Texel(uint(x1), uint(y0));
    const float4 c01 = Texel(uint(x0), uint(y1));
    const float4 c11 = Texel(uint(x1), uint(y1));
    return (c00 * (1.0f - tx) + c10 * tx) * (1.0f - ty) + (c01 * (1.0f - tx) + c11 * tx) * ty;
  }
};

// ---------------------------------------------------------------------------
// Scene data shared by every generated kernel translation unit. One instance
// lives on the host (CPU rendering, CUDA setup) and one in CUDA device memory.
// ---------------------------------------------------------------------------

struct KernelContext {
  // Images
  float4 *accumulated_color{nullptr};   // width * height
  float *accumulated_samples{nullptr};  // width * height
  uint image_width{0};
  uint image_height{0};

  // Render settings (HLSL RenderSettings layout)
  const uint8_t *render_settings{nullptr};

  // Buffers. data_buffers points to an array of data_buffer_count byte
  // pointers immediately followed by an array of data_buffer_count sizes.
  const uint *sobol_table{nullptr};  // 65536 * 1024 direction uints
  const uint8_t *camera_data{nullptr};
  const uint8_t *const *data_buffers{nullptr};
  uint data_buffer_count{0};
  const uint8_t *instance_metadatas{nullptr};
  const uint8_t *light_selector_data{nullptr};
  const uint8_t *light_metadatas{nullptr};
  const uint8_t *software_instances{nullptr};
  const uint8_t *software_nodes{nullptr};

  // Textures
  const Texture2D *sdr_textures{nullptr};
  uint sdr_texture_count{0};
  const Texture2D *hdr_textures{nullptr};
  uint hdr_texture_count{0};
};

#ifdef __CUDACC__
__constant__ KernelContext g_ctx;
#define SPARKIUM_CTX (&sparkium_portable::g_ctx)
#else
extern const KernelContext *g_ctx_ptr;
#define SPARKIUM_CTX (g_ctx_ptr)
#endif

// ---------------------------------------------------------------------------
// HLSL-compatible global resource accessors
// ---------------------------------------------------------------------------

SPARKIUM_KERNEL inline ByteAddressBuffer ResourceDataBuffer(uint index) {
  const KernelContext *c = SPARKIUM_CTX;
  if (index >= c->data_buffer_count)
    return ByteAddressBuffer{nullptr, 0};
  const size_t *sizes = reinterpret_cast<const size_t *>(c->data_buffers + c->data_buffer_count);
  return ByteAddressBuffer{c->data_buffers[index], sizes[index]};
}
// HLSL `data_buffers[i]` array syntax.
struct DataBufferTable {
  SPARKIUM_KERNEL ByteAddressBuffer operator[](uint index) const {
    const KernelContext *c = SPARKIUM_CTX;
    if (index >= c->data_buffer_count)
      return ByteAddressBuffer{nullptr, 0};
    const size_t *sizes = reinterpret_cast<const size_t *>(c->data_buffers + c->data_buffer_count);
    return ByteAddressBuffer{c->data_buffers[index], sizes[index]};
  }
};
SPARKIUM_KERNEL inline DataBufferTable ResourceDataBuffer() {
  return DataBufferTable{};
}

SPARKIUM_KERNEL inline ByteAddressBuffer ResourceSobol() {
  return ByteAddressBuffer{reinterpret_cast<const uint8_t *>(SPARKIUM_CTX->sobol_table),
                           65536ull * 1024ull * 4ull};
}
SPARKIUM_KERNEL inline ByteAddressBuffer ResourceCameraData() {
  return ByteAddressBuffer{SPARKIUM_CTX->camera_data, 160};
}
SPARKIUM_KERNEL inline ByteAddressBuffer ResourceInstanceMetadatas() {
  return ByteAddressBuffer{SPARKIUM_CTX->instance_metadatas, 1ull << 30};
}
SPARKIUM_KERNEL inline ByteAddressBuffer ResourceLightSelector() {
  return ByteAddressBuffer{SPARKIUM_CTX->light_selector_data, 1ull << 30};
}
SPARKIUM_KERNEL inline ByteAddressBuffer ResourceLightMetadatas() {
  return ByteAddressBuffer{SPARKIUM_CTX->light_metadatas, 1ull << 30};
}
SPARKIUM_KERNEL inline ByteAddressBuffer ResourceSoftwareInstances() {
  return ByteAddressBuffer{SPARKIUM_CTX->software_instances, 1ull << 30};
}
SPARKIUM_KERNEL inline ByteAddressBuffer ResourceSoftwareNodes() {
  return ByteAddressBuffer{SPARKIUM_CTX->software_nodes, 1ull << 30};
}

// Film accumulation textures (RWTexture2D semantics, indexed by uint2 pixel).
struct AccumColorView {
  struct Ref {
    float4 *ptr;
    SPARKIUM_KERNEL operator float4() const { return *ptr; }
    SPARKIUM_KERNEL Ref &operator=(const float4 &v) {
      *ptr = v;
      return *this;
    }
  };
  SPARKIUM_KERNEL Ref operator[](const uint2 &pixel) const {
    const KernelContext *c = SPARKIUM_CTX;
    return Ref{&c->accumulated_color[pixel.y * c->image_width + pixel.x]};
  }
};
struct AccumSamplesView {
  struct Ref {
    float *ptr;
    SPARKIUM_KERNEL operator float() const { return *ptr; }
    SPARKIUM_KERNEL Ref &operator=(float v) {
      *ptr = v;
      return *this;
    }
  };
  SPARKIUM_KERNEL Ref operator[](const uint2 &pixel) const {
    const KernelContext *c = SPARKIUM_CTX;
    return Ref{&c->accumulated_samples[pixel.y * c->image_width + pixel.x]};
  }
};
SPARKIUM_KERNEL inline AccumColorView ResourceAccumulatedColor() {
  return AccumColorView{};
}
SPARKIUM_KERNEL inline AccumSamplesView ResourceAccumulatedSamples() {
  return AccumSamplesView{};
}

// RenderSettings mirrored from shaders/common.hlsli (the transpiled
// definition is suppressed under SPARKIUM_PORTABLE; the byte layout must
// match Settings::RayTracing followed by Film::Info uploaded by the GPU
// pipelines).
struct RenderSettings {
  int samples_per_dispatch;
  int max_bounces;
  int alpha_shadow;
  int _settings_padding;
  float3 background_color;
  int accumulated_samples;
  float persistence;
  float clamping;
  float max_exposure;
  int view_transform;
  float exposure;
  float gamma;
  float contrast;
};

SPARKIUM_KERNEL inline const RenderSettings &ResourceRenderSettings() {
  return *reinterpret_cast<const RenderSettings *>(SPARKIUM_CTX->render_settings);
}

SPARKIUM_KERNEL inline float4 SampleTexture(int texture_index, float2 uv) {
  const KernelContext *c = SPARKIUM_CTX;
  if (texture_index & 0x1000000) {
    const uint index = uint(texture_index) & 0xFFFFFFu;
    if (index < c->hdr_texture_count)
      return c->hdr_textures[index].SampleBilinear(float2(uv.x, 1.0f - uv.y));
    return float4(1.0f);
  }
  if (uint(texture_index) < c->sdr_texture_count)
    return c->sdr_textures[uint(texture_index)].SampleBilinear(float2(uv.x, 1.0f - uv.y));
  return float4(1.0f);
}

}  // namespace sparkium_portable
