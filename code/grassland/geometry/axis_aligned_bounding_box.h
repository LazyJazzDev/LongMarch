#pragma once
#include "grassland/geometry/geometry_util.h"

namespace grassland::geometry {
template <typename Scalar, int dim>
struct AxisAlignedBoundingBox;

template <typename Scalar>
struct AxisAlignedBoundingBox<Scalar, 3> {
  Vector3<Scalar> min_bound;
  Vector3<Scalar> max_bound;

  AxisAlignedBoundingBox()
      : min_bound(Vector3<Scalar>{std::numeric_limits<Scalar>::max(),
                                  std::numeric_limits<Scalar>::max(),
                                  std::numeric_limits<Scalar>::max()}),
        max_bound(Vector3<Scalar>{std::numeric_limits<Scalar>::lowest(),
                                  std::numeric_limits<Scalar>::lowest(),
                                  std::numeric_limits<Scalar>::lowest()}) {
  }

  AxisAlignedBoundingBox(const Vector3<Scalar> &point)
      : max_bound(point), min_bound(point) {
  }

  void Expand(const Vector3<Scalar> &point) {
    min_bound = min_bound.cwiseMin(point);
    max_bound = max_bound.cwiseMax(point);
  }

  void Expand(const AxisAlignedBoundingBox<Scalar, 3> &aabb) {
    min_bound = min_bound.cwiseMin(aabb.min_bound);
    max_bound = max_bound.cwiseMax(aabb.max_bound);
  }

  Vector3<Scalar> Center() const {
    return (min_bound + max_bound) / 2;
  }

  Vector3<Scalar> Size() const {
    return max_bound - min_bound;
  }
};

template <typename Scalar>
struct AxisAlignedBoundingBox<Scalar, 2> {
  Vector2<Scalar> min_bound;
  Vector2<Scalar> max_bound;

  AxisAlignedBoundingBox()
      : min_bound(Vector2<Scalar>{std::numeric_limits<Scalar>::max(),
                                  std::numeric_limits<Scalar>::max()}),
        max_bound(Vector2<Scalar>{std::numeric_limits<Scalar>::lowest(),
                                  std::numeric_limits<Scalar>::lowest()}) {
  }

  AxisAlignedBoundingBox(const Vector2<Scalar> &point)
      : max_bound(point), min_bound(point) {
  }

  void Expand(const Vector2<Scalar> &point) {
    min_bound = min_bound.cwiseMin(point);
    max_bound = max_bound.cwiseMax(point);
  }

  void Expand(const AxisAlignedBoundingBox<Scalar, 2> &aabb) {
    min_bound = min_bound.cwiseMin(aabb.min_bound);
    max_bound = max_bound.cwiseMax(aabb.max_bound);
  }

  Vector2<Scalar> Center() const {
    return (min_bound + max_bound) / 2;
  }

  Vector2<Scalar> Size() const {
    return max_bound - min_bound;
  }
};

template <typename Scalar>
using AxisAlignedBoundingBox2 = AxisAlignedBoundingBox<Scalar, 2>;
template <typename Scalar>
using AxisAlignedBoundingBox3 = AxisAlignedBoundingBox<Scalar, 3>;

using AxisAlignedBoundingBox2f = AxisAlignedBoundingBox2<float>;
using AxisAlignedBoundingBox2d = AxisAlignedBoundingBox2<double>;

using AxisAlignedBoundingBox3f = AxisAlignedBoundingBox3<float>;
using AxisAlignedBoundingBox3d = AxisAlignedBoundingBox3<double>;

template <typename Scalar, int dim>
using AABB = AxisAlignedBoundingBox<Scalar, dim>;
template <typename Scalar>
using AABB2 = AxisAlignedBoundingBox<Scalar, 2>;
template <typename Scalar>
using AABB3 = AxisAlignedBoundingBox<Scalar, 3>;

using AABB2f = AABB2<float>;
using AABB2d = AABB2<double>;

using AABB3f = AABB3<float>;
using AABB3d = AABB3<double>;
}  // namespace grassland::geometry
