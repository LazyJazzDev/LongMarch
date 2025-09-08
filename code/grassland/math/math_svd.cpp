#include "grassland/math/math_svd.h"

namespace CD {
template <typename Real>
LM_DEVICE_FUNC void EigenDecomp(const Matrix2<Real> &A, Matrix2<Real> &D, Matrix2<Real> &G) {
  Real a = A(0, 0);
  Real b = A(0, 1);
  Real d = A(1, 1);

  if (fabs(b) < Eps<Real>()) {
    D = A;
    G.setIdentity();
    return;
  }

  Vector2<Real> cs2;
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

template LM_DEVICE_FUNC void EigenDecomp(const Matrix2<float> &A, Matrix2<float> &D, Matrix2<float> &G);
template LM_DEVICE_FUNC void EigenDecomp(const Matrix2<double> &A, Matrix2<double> &D, Matrix2<double> &G);

template <typename Real>
LM_DEVICE_FUNC void SVD(const Matrix<Real, 3, 2> &A, Matrix<Real, 3, 2> &U, Matrix2<Real> &S, Matrix2<Real> &Vt) {
  Matrix<Real, 2, 2> AtA = A.transpose() * A;
  EigenDecomp(AtA, S, Vt);
  S(0, 0) = std::sqrt(S(0, 0));
  S(1, 1) = std::sqrt(S(1, 1));
  U = A * Vt * S.inverse();
  Vt(0, 1) = -Vt(0, 1);
  Vt(1, 0) = -Vt(1, 0);
}

template LM_DEVICE_FUNC void SVD(const Matrix<float, 3, 2> &A,
                                 Matrix<float, 3, 2> &U,
                                 Matrix2<float> &S,
                                 Matrix2<float> &Vt);
template LM_DEVICE_FUNC void SVD(const Matrix<double, 3, 2> &A,
                                 Matrix<double, 3, 2> &U,
                                 Matrix2<double> &S,
                                 Matrix2<double> &Vt);

template <typename Real>
LM_DEVICE_FUNC void SVD(const Matrix2<Real> &A, Matrix2<Real> &U, Matrix2<Real> &S, Matrix2<Real> &Vt) {
  Matrix2<Real> AtA = A.transpose() * A;
  EigenDecomp(AtA, S, Vt);
  S(0, 0) = std::sqrt(S(0, 0));
  S(1, 1) = std::sqrt(S(1, 1));
  U = A * Vt * S.inverse();
  Vt(0, 1) = -Vt(0, 1);
  Vt(1, 0) = -Vt(1, 0);
}

template LM_DEVICE_FUNC void SVD(const Matrix2<float> &A, Matrix2<float> &U, Matrix2<float> &S, Matrix2<float> &Vt);
template LM_DEVICE_FUNC void SVD(const Matrix2<double> &A, Matrix2<double> &U, Matrix2<double> &S, Matrix2<double> &Vt);

}  // namespace CD
