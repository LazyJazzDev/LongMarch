#pragma once
#include "grassland/math/math_util.h"

namespace CD {

template <typename Scalar>
LM_DEVICE_FUNC Scalar PolygonArea(const Vector2<Scalar> *vertices, int n) {
  Scalar area = 0;
  for (int i = 0; i < n; i++) {
    const auto &p0 = vertices[i];
    const auto &p1 = vertices[(i + 1) % n];
    area += p0[0] * p1[1] - p1[0] * p0[1];
  }
  return area / 2;
}

template <typename Scalar>
LM_DEVICE_FUNC Scalar TetrahedronVolume(const Vector3<Scalar> &p0,
                                        const Vector3<Scalar> &p1,
                                        const Vector3<Scalar> &p2,
                                        const Vector3<Scalar> &p3) {
  return (p1 - p0).cross(p2 - p0).dot(p3 - p0) / 6;
}

template <typename Scalar>
LM_DEVICE_FUNC Scalar SolidAngle(const Vector3<Scalar> &v0, const Vector3<Scalar> &v1, const Vector3<Scalar> &v2) {
  Scalar v0_norm = v0.norm();
  Scalar v1_norm = v1.norm();
  Scalar v2_norm = v2.norm();
  return 2 * atan(v0.dot(v1.cross(v2)) /
                  (v0_norm * v1_norm * v2_norm + v0.dot(v1) * v2_norm + v1.dot(v2) * v0_norm + v2.dot(v0) * v1_norm));
}

}  // namespace CD
