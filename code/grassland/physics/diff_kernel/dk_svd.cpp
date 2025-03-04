#include "grassland/physics/diff_kernel/dk_svd.h"

namespace grassland {
template <typename Real>
LM_DEVICE_FUNC void EigenDecomp(const Eigen::Matrix2<Real> &A, Eigen::Matrix2<Real> &D, Eigen::Matrix2<Real> &G) {
  Real a = A(0, 0);
  Real b = A(0, 1);
  Real d = A(1, 1);

  if (fabs(b) < Eps<Real>()) {
    D = A;
    G.setIdentity();
    return;
  }

  Eigen::Vector2<Real> cs2;
  cs2 << (a - d) / 2, -b;
  cs2.normalize();
  if (cs2[0] < 0) {
    cs2 = -cs2;
  }
  Real c = sqrt(2 + 2 * cs2[0]) / 2;
  Real s = sqrt(2 - 2 * cs2[0]) / 2;
  if (cs2[1] < 0) {
    s = -s;
  }
  G << c, -s, s, c;
  D = G * A * G.transpose();
  G(0, 1) = -G(0, 1);
  G(1, 0) = -G(1, 0);
}

template LM_DEVICE_FUNC void EigenDecomp(const Eigen::Matrix2<float> &A,
                                         Eigen::Matrix2<float> &D,
                                         Eigen::Matrix2<float> &G);
template LM_DEVICE_FUNC void EigenDecomp(const Eigen::Matrix2<double> &A,
                                         Eigen::Matrix2<double> &D,
                                         Eigen::Matrix2<double> &G);

template <typename Real>
LM_DEVICE_FUNC void SVD(const Eigen::Matrix<Real, 3, 2> &A,
                        Eigen::Matrix<Real, 3, 2> &U,
                        Eigen::Matrix2<Real> &S,
                        Eigen::Matrix2<Real> &Vt) {
  Eigen::Matrix<Real, 2, 2> AtA = A.transpose() * A;
  EigenDecomp(AtA, S, Vt);
  S(0, 0) = std::sqrt(S(0, 0));
  S(1, 1) = std::sqrt(S(1, 1));
  U = A * Vt * S.inverse();
  Vt(0, 1) = -Vt(0, 1);
  Vt(1, 0) = -Vt(1, 0);
}

template void SVD(const Eigen::Matrix<float, 3, 2> &A,
                  Eigen::Matrix<float, 3, 2> &U,
                  Eigen::Matrix2<float> &S,
                  Eigen::Matrix2<float> &Vt);
template void SVD(const Eigen::Matrix<double, 3, 2> &A,
                  Eigen::Matrix<double, 3, 2> &U,
                  Eigen::Matrix2<double> &S,
                  Eigen::Matrix2<double> &Vt);
}  // namespace grassland
