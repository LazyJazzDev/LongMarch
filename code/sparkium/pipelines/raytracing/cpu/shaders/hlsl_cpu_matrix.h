// HLSL matrix types for the Sparkium CPU backend.
//
// The shaders depend on HLSL's matrix conventions, not glm's, so the types here
// store rows and index rows:
//
//   * `m[i]` is row i. buffer_helper.hlsli builds matrices with
//     `mat[0] = LoadFloat4(...)` and then transposes them, which only yields
//     the intended layout under row indexing.
//   * `mul(A, B)` multiplies A by B the way HLSL does. For a matrix times a
//     vector that means the result has as many components as the matrix has
//     rows, which is why `mul(float3x4, float4)` is a float3.
#pragma once

#include <type_traits>

#include "sparkium/pipelines/raytracing/cpu/shaders/hlsl_cpu_ops.h"

namespace sparkium::cpu::hlsl {

// A pack expansion cannot be an argument to an alias template, so the matrix
// constructor's constraint goes through a dedicated trait.
template <class Target, class... Args>
struct AllConvertibleTo;
template <class Target>
struct AllConvertibleTo<Target> : std::true_type {};
template <class Target, class First, class... Rest>
struct AllConvertibleTo<Target, First, Rest...>
    : std::conditional_t<std::is_convertible_v<First, Target>, AllConvertibleTo<Target, Rest...>,
                         std::false_type> {};

// `RowVector` fixes both the column count and the element type; R is the row
// count. So float3x4 is Mat<3, Float4> and float4x3 is Mat<4, Float3>.
template <int R, class RowVector>
struct Mat {
  using value_type = ValueTypeOf<RowVector>;
  static constexpr int row_count = R;
  static constexpr int column_count = components_v<RowVector>;

  RowVector rows[R];

  Mat() : rows{} {
  }

  // HLSL builds a matrix from its rows: `float3x3(T, B, N)`.
  template <class... RowArgs,
            std::enable_if_t<(sizeof...(RowArgs) > 1) && sizeof...(RowArgs) == R &&
                                 AllConvertibleTo<RowVector, RowArgs...>::value,
                             int> = 0>
  Mat(RowArgs... args) : rows{RowVector(args)...} {
  }

  // HLSL truncates rows when a taller matrix is assigned to a shorter one.
  template <int OtherR, std::enable_if_t<(OtherR > R), int> = 0>
  Mat(const Mat<OtherR, RowVector> &other) {
    for (int i = 0; i < R; ++i)
      rows[i] = other[i];
  }

  RowVector &operator[](int i) {
    return rows[i];
  }
  const RowVector &operator[](int i) const {
    return rows[i];
  }
};

using Float2x2 = Mat<2, Float2>;
using Float2x4 = Mat<2, Float4>;
using Float3x3 = Mat<3, Float3>;
using Float3x4 = Mat<3, Float4>;
using Float4x2 = Mat<4, Float2>;
using Float4x3 = Mat<4, Float3>;
using Float4x4 = Mat<4, Float4>;

template <int R, class RowVector>
struct Traits<Mat<R, RowVector>> {
  static constexpr bool is_vector = false;
  static constexpr bool is_matrix = true;
  static constexpr int components = R * components_v<RowVector>;
  using value_type = ValueTypeOf<RowVector>;
};

template <int R, class RowVector>
Mat<R, RowVector> operator*(const Mat<R, RowVector> &m, ValueTypeOf<RowVector> s) {
  Mat<R, RowVector> result;
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < Mat<R, RowVector>::column_count; ++j)
      result[i][j] = static_cast<ValueTypeOf<RowVector>>(m[i][j] * s);
  return result;
}

template <int R, class RowVector>
Mat<R, RowVector> operator*(ValueTypeOf<RowVector> s, const Mat<R, RowVector> &m) {
  return m * s;
}

// Swaps rows and columns: transpose(float3x4) is a float4x3.
template <int R, class RowVector>
Mat<Mat<R, RowVector>::column_count, VecOfT<R, ValueTypeOf<RowVector>>> transpose(
    const Mat<R, RowVector> &m) {
  using ResultRow = VecOfT<R, ValueTypeOf<RowVector>>;
  Mat<Mat<R, RowVector>::column_count, ResultRow> result;
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < Mat<R, RowVector>::column_count; ++j)
      result[j][i] = static_cast<ValueTypeOf<RowVector>>(m[i][j]);
  return result;
}

// `mul(A, B)` follows HLSL: matrix times column vector, row vector times
// matrix, and matrix times matrix.
template <int R, class RowVector, class V,
          std::enable_if_t<is_vector_v<V> && components_v<V> == Mat<R, RowVector>::column_count,
                           int> = 0>
VecOfT<R, ValueTypeOf<RowVector>> mul(const Mat<R, RowVector> &m, const V &v) {
  VecOfT<R, ValueTypeOf<RowVector>> result;
  for (int i = 0; i < R; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < Mat<R, RowVector>::column_count; ++j)
      sum += static_cast<float>(m[i][j]) * static_cast<float>(v[j]);
    result[i] = static_cast<ValueTypeOf<RowVector>>(sum);
  }
  return result;
}

template <class V, int R, class RowVector,
          std::enable_if_t<is_vector_v<V> && components_v<V> == R, int> = 0>
VecOfT<Mat<R, RowVector>::column_count, ValueTypeOf<RowVector>> mul(const V &v,
                                                                   const Mat<R, RowVector> &m) {
  VecOfT<Mat<R, RowVector>::column_count, ValueTypeOf<RowVector>> result;
  for (int j = 0; j < Mat<R, RowVector>::column_count; ++j) {
    float sum = 0.0f;
    for (int i = 0; i < R; ++i)
      sum += static_cast<float>(v[i]) * static_cast<float>(m[i][j]);
    result[j] = static_cast<ValueTypeOf<RowVector>>(sum);
  }
  return result;
}

template <int R, class RowVectorA, int Inner, class RowVectorB,
          std::enable_if_t<Mat<R, RowVectorA>::column_count == Inner, int> = 0>
Mat<R, VecOfT<Mat<Inner, RowVectorB>::column_count, ValueTypeOf<RowVectorB>>> mul(
    const Mat<R, RowVectorA> &a,
    const Mat<Inner, RowVectorB> &b) {
  using ResultRow = VecOfT<Mat<Inner, RowVectorB>::column_count, ValueTypeOf<RowVectorB>>;
  Mat<R, ResultRow> result;
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < Mat<Inner, RowVectorB>::column_count; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < Inner; ++k)
        sum += static_cast<float>(a[i][k]) * static_cast<float>(b[k][j]);
      result[i][j] = static_cast<ValueTypeOf<RowVectorB>>(sum);
    }
  return result;
}

// HLSL spells these floatNxM; keep the names the shaders use.
using float2x2 = Float2x2;
using float2x4 = Float2x4;
using float3x3 = Float3x3;
using float3x4 = Float3x4;
using float4x2 = Float4x2;
using float4x3 = Float4x3;
using float4x4 = Float4x4;

}  // namespace sparkium::cpu::hlsl
