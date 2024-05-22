#pragma once
#include "Eigen/Eigen"
#include "grassland/utils/utils.h"

namespace grassland::geometry {
using Eigen::Matrix;
using Eigen::Matrix2;
using Eigen::Matrix3;
using Eigen::Matrix4;
using Eigen::RowVector2;
using Eigen::RowVector3;
using Eigen::RowVector4;
using Eigen::Vector;
using Eigen::Vector2;
using Eigen::Vector3;
using Eigen::Vector4;

template <typename Scalar>
Scalar Eps();

template <>
inline float Eps<float>() {
  return 1e-4f;
}

template <>
inline double Eps<double>() {
  return 1e-8;
}

template <typename Scalar>
int Sign(Scalar x) {
  if (x < 0) {
    return -1;
  } else if (x > 0) {
    return 1;
  } else {
    return 0;
  }
}

}  // namespace grassland::geometry
