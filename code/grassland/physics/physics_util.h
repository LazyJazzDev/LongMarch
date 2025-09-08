#pragma once
#include <Eigen/Eigen>
#include <iostream>

#include "grassland/math/math.h"
#include "grassland/util/util.h"

namespace CD {

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix3<Real> Skew3(const Eigen::Matrix<Real, 3, 1> &v) {
  Eigen::Matrix3<Real> m;
  m << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
  return m;
}

}  // namespace CD
