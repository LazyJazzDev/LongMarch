#pragma once
#include "grassland/math/math_util.h"

namespace grassland {
template <typename Scalar, int dim>
struct AxisAlignedBoundingBox;

template <typename Scalar>
struct AxisAlignedBoundingBox<Scalar, 3> {
  Vector3<Scalar> lower_bound;
  Vector3<Scalar> upper_bound;

  AxisAlignedBoundingBox()
      : lower_bound(Vector3<Scalar>{std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::max(),
                                    std::numeric_limits<Scalar>::max()}),
        upper_bound(Vector3<Scalar>{std::numeric_limits<Scalar>::lowest(), std::numeric_limits<Scalar>::lowest(),
                                    std::numeric_limits<Scalar>::lowest()}) {
  }

  AxisAlignedBoundingBox(const Vector3<Scalar> &point) : upper_bound(point), lower_bound(point) {
  }

  void Expand(const Vector3<Scalar> &point) {
    lower_bound = lower_bound.cwiseMin(point);
    upper_bound = upper_bound.cwiseMax(point);
  }

  void Expand(const AxisAlignedBoundingBox<Scalar, 3> &aabb) {
    lower_bound = lower_bound.cwiseMin(aabb.lower_bound);
    upper_bound = upper_bound.cwiseMax(aabb.upper_bound);
  }

  Vector3<Scalar> Center() const {
    return (lower_bound + upper_bound) / 2;
  }

  Vector3<Scalar> Size() const {
    return upper_bound - lower_bound;
  }

  bool Contain(const Vector3<Scalar> &point) const {
    return (point[0] >= lower_bound[0] && point[0] <= upper_bound[0] && point[1] >= lower_bound[1] &&
            point[1] <= upper_bound[1] && point[2] >= lower_bound[2] && point[2] <= upper_bound[2]);
  }
};

template <typename Scalar>
struct AxisAlignedBoundingBox<Scalar, 2> {
  Vector2<Scalar> lower_bound;
  Vector2<Scalar> upper_bound;

  AxisAlignedBoundingBox()
      : lower_bound(Vector2<Scalar>{std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::max()}),
        upper_bound(Vector2<Scalar>{std::numeric_limits<Scalar>::lowest(), std::numeric_limits<Scalar>::lowest()}) {
  }

  AxisAlignedBoundingBox(const Vector2<Scalar> &point) : upper_bound(point), lower_bound(point) {
  }

  void Expand(const Vector2<Scalar> &point) {
    lower_bound = lower_bound.cwiseMin(point);
    upper_bound = upper_bound.cwiseMax(point);
  }

  void Expand(const AxisAlignedBoundingBox<Scalar, 2> &aabb) {
    lower_bound = lower_bound.cwiseMin(aabb.lower_bound);
    upper_bound = upper_bound.cwiseMax(aabb.upper_bound);
  }

  Vector2<Scalar> Center() const {
    return (lower_bound + upper_bound) / 2;
  }

  Vector2<Scalar> Size() const {
    return upper_bound - lower_bound;
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

using AABB = AxisAlignedBoundingBox3f;

}  // namespace grassland
