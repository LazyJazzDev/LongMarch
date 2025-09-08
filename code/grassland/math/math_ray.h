#pragma once
#include "grassland/math/math_util.h"

namespace CD {
template <typename Scalar, int dim>
struct Ray;

template <typename Scalar>
struct Ray<Scalar, 3> {
  Vector3<Scalar> origin;
  Vector3<Scalar> direction;
};

template <typename Scalar>
struct Ray<Scalar, 2> {
  Vector2<Scalar> origin;
  Vector2<Scalar> direction;
};

template <typename Scalar>
using Ray2 = Ray<Scalar, 2>;
template <typename Scalar>
using Ray3 = Ray<Scalar, 3>;

using Ray2f = Ray2<float>;
using Ray2d = Ray2<double>;

using Ray3f = Ray3<float>;
using Ray3d = Ray3<double>;
}  // namespace CD
