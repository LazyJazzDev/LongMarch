#pragma once
#include "grassland/geometry/common.h"

namespace grassland::geometry {

template <typename Scalar>
Scalar PolygonArea(const Vector2<Scalar> *vertices, int n) {
  Scalar area = 0;
  for (int i = 0; i < n; i++) {
    const auto &p0 = vertices[i];
    const auto &p1 = vertices[(i + 1) % n];
    area += p0[0] * p1[1] - p1[0] * p0[1];
  }
  return area / 2;
}

}  // namespace grassland::geometry
