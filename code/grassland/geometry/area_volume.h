#pragma once
#include "grassland/geometry/geometry_util.h"

namespace grassland::geometry {

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

}  // namespace grassland::geometry
