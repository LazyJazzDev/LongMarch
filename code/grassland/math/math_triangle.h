#pragma once
#include "grassland/math/math_util.h"

namespace CD {
template <typename Scalar, int dim>
struct Triangle;

template <typename Scalar>
struct Triangle<Scalar, 3> {
  Matrix<Scalar, 3, 3> m;

  LM_DEVICE_FUNC auto operator[](int i) {
    return m.col(i);
  }

  LM_DEVICE_FUNC auto operator[](int i) const {
    return m.col(i);
  }

  LM_DEVICE_FUNC Vector3<Scalar> normal() const {
    return (m.col(1) - m.col(0)).cross(m.col(2) - m.col(0)).normalized();
  }

  LM_DEVICE_FUNC Vector3<Scalar> vertex(int i) const {
    return m.col(i);
  }
};

template <typename Scalar>
struct Triangle<Scalar, 2> {
  Matrix<Scalar, 2, 3> m;

  LM_DEVICE_FUNC auto operator[](int i) {
    return m.col(i);
  }

  LM_DEVICE_FUNC auto operator[](int i) const {
    return m.col(i);
  }
};

template <typename Scalar>
using Triangle2 = Triangle<Scalar, 2>;
template <typename Scalar>
using Triangle3 = Triangle<Scalar, 3>;

using Triangle2f = Triangle2<float>;
using Triangle2d = Triangle2<double>;

using Triangle3f = Triangle3<float>;
using Triangle3d = Triangle3<double>;
}  // namespace CD
