#pragma once
#include "cao_di/math/math_util.h"

namespace CD {

template <typename Real>
LM_DEVICE_FUNC void EigenDecomp(const Eigen::Matrix2<Real> &A, Eigen::Matrix2<Real> &D, Eigen::Matrix2<Real> &G);

template <typename Real>
LM_DEVICE_FUNC void SVD(const Eigen::Matrix<Real, 3, 2> &A,
                        Eigen::Matrix<Real, 3, 2> &U,
                        Eigen::Matrix2<Real> &S,
                        Eigen::Matrix2<Real> &Vt);

template <typename Real>
LM_DEVICE_FUNC void SVD(const Eigen::Matrix2<Real> &A,
                        Eigen::Matrix2<Real> &U,
                        Eigen::Matrix2<Real> &S,
                        Eigen::Matrix2<Real> &Vt);

}  // namespace CD
