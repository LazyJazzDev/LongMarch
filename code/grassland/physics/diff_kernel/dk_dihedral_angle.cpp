#include "grassland/physics/diff_kernel/dk_dihedral_angle.h"

namespace grassland {
template <typename Real>
LM_DEVICE_FUNC bool DihedralAngleAssistEdgesToNormalsAxis<Real>::ValidInput(const InputType &A) const {
  return A.col(0).cross(A.col(1)).norm() > algebra::Eps<Real>() * 100 &&
         A.col(1).cross(A.col(2)).norm() > algebra::Eps<Real>() * 100;
}

template <typename Real>
LM_DEVICE_FUNC typename DihedralAngleAssistEdgesToNormalsAxis<Real>::OutputType
DihedralAngleAssistEdgesToNormalsAxis<Real>::operator()(const InputType &E) const {
  CrossNormalized<Real> cross_normalized;
  VecNormalized<Real> vec_normalized;
  OutputType normals_axis;
  normals_axis.col(0) = cross_normalized(E.block(0, 0, 3, 2));
  normals_axis.col(1) = -cross_normalized(E.block(0, 1, 3, 2));
  normals_axis.col(2) = vec_normalized(E.col(1));
  return normals_axis;
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real,
                             DihedralAngleAssistEdgesToNormalsAxis<Real>::OutputSize,
                             DihedralAngleAssistEdgesToNormalsAxis<Real>::InputSize>
DihedralAngleAssistEdgesToNormalsAxis<Real>::Jacobian(const InputType &E) const {
  Eigen::Matrix<Real, OutputSize, InputSize> J;
  J.setZero();
  CrossNormalized<Real> cross_normalized;
  VecNormalized<Real> vec_normalized;
  J.block(0, 0, 3, 6) = cross_normalized.Jacobian(E.block(0, 0, 3, 2));
  J.block(3, 3, 3, 6) = -cross_normalized.Jacobian(E.block(0, 1, 3, 2));
  J.block(6, 3, 3, 3) = vec_normalized.Jacobian(E.col(1));
  return J;
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real,
                             DihedralAngleAssistEdgesToNormalsAxis<Real>::OutputSize,
                             DihedralAngleAssistEdgesToNormalsAxis<Real>::InputSize>
DihedralAngleAssistEdgesToNormalsAxis<Real>::Hessian(const InputType &E) const {
  HessianTensor<Real, OutputSize, InputSize> H;
  CrossNormalized<Real> cross_normalized;
  VecNormalized<Real> vec_normalized;
  auto cross_normalized_hessian = cross_normalized.Hessian(E.block(0, 0, 3, 2));
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 6; j++) {
      for (int k = 0; k < 6; k++) {
        H(i, j, k) = cross_normalized_hessian(i, j, k);
      }
    }
  }
  cross_normalized_hessian = -cross_normalized.Hessian(E.block(0, 1, 3, 2));
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 6; j++) {
      for (int k = 0; k < 6; k++) {
        H(i + 3, j + 3, k + 3) = cross_normalized_hessian(i, j, k);
      }
    }
  }
  auto vec_normalized_hessian = vec_normalized.Hessian(E.col(1));
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        H(i + 6, j + 3, k + 3) = vec_normalized_hessian(i, j, k);
      }
    }
  }
  return H;
}

template class DihedralAngleAssistEdgesToNormalsAxis<float>;
template class DihedralAngleAssistEdgesToNormalsAxis<double>;

template <typename Real>
LM_DEVICE_FUNC bool DihedralAngleAssistNormalsAxisToSinCosTheta<Real>::ValidInput(const InputType &normals_axis) const {
  return true;
}

template <typename Real>
LM_DEVICE_FUNC typename DihedralAngleAssistNormalsAxisToSinCosTheta<Real>::OutputType
DihedralAngleAssistNormalsAxisToSinCosTheta<Real>::operator()(const InputType &N) const {
  Determinant3<Real> det3;
  Dot<Real> dot;
  return OutputType{det3(N).value(), dot(N.col(0), N.col(1)).value()};
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real,
                             DihedralAngleAssistNormalsAxisToSinCosTheta<Real>::OutputSize,
                             DihedralAngleAssistNormalsAxisToSinCosTheta<Real>::InputSize>
DihedralAngleAssistNormalsAxisToSinCosTheta<Real>::Jacobian(const InputType &N) const {
  Eigen::Matrix<Real, OutputSize, InputSize> J;
  J.setZero();
  Determinant3<Real> det3;
  Dot<Real> dot;
  J.row(0) = det3.Jacobian(N);
  J.block(1, 0, 1, 6) = dot.Jacobian(N.block(0, 0, 3, 2));
  return J;
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real,
                             DihedralAngleAssistNormalsAxisToSinCosTheta<Real>::OutputSize,
                             DihedralAngleAssistNormalsAxisToSinCosTheta<Real>::InputSize>
DihedralAngleAssistNormalsAxisToSinCosTheta<Real>::Hessian(const InputType &A) const {
  HessianTensor<Real, OutputSize, InputSize> H;
  H.m[0] = Determinant3<Real>().Hessian(A).m[0];
  H.m[1].block(0, 0, 6, 6) = Dot<Real>().Hessian(A.block(0, 0, 3, 2)).m[0];
  return H;
}

template class DihedralAngleAssistNormalsAxisToSinCosTheta<float>;
template class DihedralAngleAssistNormalsAxisToSinCosTheta<double>;

template <typename Real>
LM_DEVICE_FUNC bool DihedralAngleAssistVerticesToEdges<Real>::ValidInput(const InputType &) const {
  return true;
}

template <typename Real>
LM_DEVICE_FUNC typename DihedralAngleAssistVerticesToEdges<Real>::OutputType
DihedralAngleAssistVerticesToEdges<Real>::operator()(const InputType &V) const {
  OutputType res;
  res.col(0) = V.col(1) - V.col(0);
  res.col(1) = V.col(2) - V.col(1);
  res.col(2) = V.col(3) - V.col(2);
  return res;
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real,
                             DihedralAngleAssistVerticesToEdges<Real>::OutputSize,
                             DihedralAngleAssistVerticesToEdges<Real>::InputSize>
DihedralAngleAssistVerticesToEdges<Real>::Jacobian(const InputType &) const {
  Eigen::Matrix<Real, OutputSize, InputSize> J;
  J.setZero();
  // J.block<3, 3>(0, 0) = -Eigen::Matrix<Real, 3, 3>::Identity();
  // J.block<3, 3>(0, 3) = Eigen::Matrix<Real, 3, 3>::Identity();
  // J.block<3, 3>(3, 3) = -Eigen::Matrix<Real, 3, 3>::Identity();
  // J.block<3, 3>(3, 6) = Eigen::Matrix<Real, 3, 3>::Identity();
  // J.block<3, 3>(6, 6) = -Eigen::Matrix<Real, 3, 3>::Identity();
  // J.block<3, 3>(6, 9) = Eigen::Matrix<Real, 3, 3>::Identity();
  for (int i = 0; i < 3; i++) {
    J(i, i) = -1;
    J(i, i + 3) = 1;
    J(i + 3, i + 3) = -1;
    J(i + 3, i + 6) = 1;
    J(i + 6, i + 6) = -1;
    J(i + 6, i + 9) = 1;
  }
  return J;
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real,
                             DihedralAngleAssistVerticesToEdges<Real>::OutputSize,
                             DihedralAngleAssistVerticesToEdges<Real>::InputSize>
DihedralAngleAssistVerticesToEdges<Real>::Hessian(const InputType &) const {
  HessianTensor<Real, OutputSize, InputSize> H;
  return H;
}

template class DihedralAngleAssistVerticesToEdges<float>;
template class DihedralAngleAssistVerticesToEdges<double>;

template <typename Real>
LM_DEVICE_FUNC bool DihedralEnergy<Real>::ValidInput(const InputType &) const {
  return true;
}

template <typename Real>
LM_DEVICE_FUNC typename DihedralEnergy<Real>::OutputType DihedralEnergy<Real>::operator()(const InputType &V) const {
  DihedralAngleByVertices<Real> dihedral_angle;
  Real res = dihedral_angle(V).value();
  res -= rest_angle;
  return OutputType{res * res};
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, DihedralEnergy<Real>::OutputSize, DihedralEnergy<Real>::InputSize>
DihedralEnergy<Real>::Jacobian(const InputType &V) const {
  DihedralAngleByVertices<Real> dihedral_angle;
  auto angle = dihedral_angle(V).value() - rest_angle;
  auto J = dihedral_angle.Jacobian(V);
  return (Real(2.0) * angle) * J;
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, DihedralEnergy<Real>::OutputSize, DihedralEnergy<Real>::InputSize>
DihedralEnergy<Real>::Hessian(const InputType &V) const {
  DihedralAngleByVertices<Real> dihedral_angle;
  auto angle = dihedral_angle(V).value() - rest_angle;
  auto J = dihedral_angle.Jacobian(V);
  auto H = dihedral_angle.Hessian(V);
  H.m[0] = H.m[0] * angle + J.transpose() * J;
  return H * 2.0;
}

template class DihedralEnergy<float>;
template class DihedralEnergy<double>;

template <typename Real>
LM_DEVICE_FUNC bool DihedralAngle<Real>::ValidInput(const InputType &) const {
  return true;
}

template <typename Real>
LM_DEVICE_FUNC typename DihedralAngle<Real>::OutputType DihedralAngle<Real>::operator()(const InputType &V) const {
  Eigen::Vector3<Real> e0 = V.col(2) - V.col(1);
  Eigen::Vector3<Real> e1 = V.col(0) - V.col(2);
  Eigen::Vector3<Real> e2 = V.col(0) - V.col(1);
  Eigen::Vector3<Real> e1_tilda = V.col(3) - V.col(2);
  Eigen::Vector3<Real> e2_tilda = V.col(3) - V.col(1);
  Eigen::Vector3<Real> n = e2.cross(e1).normalized();
  Eigen::Vector3<Real> n_tilda = e1_tilda.cross(e2_tilda).normalized();
  Real sin_theta = n.cross(n_tilda).dot(e0.normalized());
  Real cos_theta = n.dot(n_tilda);
  return OutputType{atan2(sin_theta, cos_theta)};
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, DihedralAngle<Real>::OutputSize, DihedralAngle<Real>::InputSize>
DihedralAngle<Real>::Jacobian(const InputType &V) const {
  Eigen::Vector3<Real> e0 = V.col(2) - V.col(1);
  Eigen::Vector3<Real> e1 = V.col(0) - V.col(2);
  Eigen::Vector3<Real> e2 = V.col(0) - V.col(1);
  Eigen::Vector3<Real> e1_tilda = V.col(3) - V.col(2);
  Eigen::Vector3<Real> e2_tilda = V.col(3) - V.col(1);

  Real e0_norm_sqr = e0.squaredNorm();
  Real e1_norm_sqr = e1.squaredNorm();
  Real e2_norm_sqr = e2.squaredNorm();
  Real e1_tilda_norm_sqr = e1_tilda.squaredNorm();
  Real e2_tilda_norm_sqr = e2_tilda.squaredNorm();

  Real e0_norm = sqrt(e0_norm_sqr);
  Real e1_norm = sqrt(e1_norm_sqr);
  Real e2_norm = sqrt(e2_norm_sqr);
  Real e1_tilda_norm = sqrt(e1_tilda_norm_sqr);
  Real e2_tilda_norm = sqrt(e2_tilda_norm_sqr);

  Eigen::Vector3<Real> m0 = -(e1 - e1.dot(e0) * e0 / e0_norm_sqr).normalized();
  Eigen::Vector3<Real> m1 = (e2 - e2.dot(e1) * e1 / e1_norm_sqr).normalized();
  Eigen::Vector3<Real> m2 = (e1 - e1.dot(e2) * e2 / e2_norm_sqr).normalized();

  Eigen::Vector3<Real> m0_tilda = -(e1_tilda - e1_tilda.dot(e0) * e0 / e0_norm_sqr).normalized();
  Eigen::Vector3<Real> m1_tilda = (e2_tilda - e2_tilda.dot(e1_tilda) * e1_tilda / e1_tilda_norm_sqr).normalized();
  Eigen::Vector3<Real> m2_tilda = (e1_tilda - e1_tilda.dot(e2_tilda) * e2_tilda / e2_tilda_norm_sqr).normalized();

  Real h0 = -m0.dot(e1);
  Real h1 = m1.dot(e2);
  Real h2 = m2.dot(e1);

  Real h0_tilda = -m0_tilda.dot(e1_tilda);
  Real h1_tilda = m1_tilda.dot(e2_tilda);
  Real h2_tilda = m2_tilda.dot(e1_tilda);

  Eigen::Vector3<Real> n = e2.cross(e1).normalized();
  Eigen::Vector3<Real> n_tilda = e1_tilda.cross(e2_tilda).normalized();

  Real cos_a0 = e1.dot(e2) / (e1_norm * e2_norm);
  Real cos_a1 = e2.dot(e0) / (e2_norm * e0_norm);
  Real cos_a2 = -e0.dot(e1) / (e0_norm * e1_norm);

  Real cos_a0_tilda = e1_tilda.dot(e2_tilda) / (e1_tilda_norm * e2_tilda_norm);
  Real cos_a1_tilda = e2_tilda.dot(e0) / (e2_tilda_norm * e0_norm);
  Real cos_a2_tilda = -e0.dot(e1_tilda) / (e0_norm * e1_tilda_norm);

  Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> J;
  J.block(0, 0, 1, 3) = -Real(1.0) / h0 * n.transpose();
  J.block(0, 3, 1, 3) = (cos_a2 / h1 * n + cos_a2_tilda / h1_tilda * n_tilda).transpose();
  J.block(0, 6, 1, 3) = (cos_a1 / h2 * n + cos_a1_tilda / h2_tilda * n_tilda).transpose();
  J.block(0, 9, 1, 3) = -Real(1.0) / h0_tilda * n_tilda.transpose();
  return J;
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, DihedralAngle<Real>::OutputSize, DihedralAngle<Real>::InputSize>
DihedralAngle<Real>::Hessian(const InputType &V) const {
  Eigen::Vector3<Real> e0 = V.col(2) - V.col(1);
  Eigen::Vector3<Real> e1 = V.col(0) - V.col(2);
  Eigen::Vector3<Real> e2 = V.col(0) - V.col(1);
  Eigen::Vector3<Real> e1_tilda = V.col(3) - V.col(2);
  Eigen::Vector3<Real> e2_tilda = V.col(3) - V.col(1);

  Real e0_norm_sqr = e0.squaredNorm();
  Real e1_norm_sqr = e1.squaredNorm();
  Real e2_norm_sqr = e2.squaredNorm();
  Real e1_tilda_norm_sqr = e1_tilda.squaredNorm();
  Real e2_tilda_norm_sqr = e2_tilda.squaredNorm();

  Real e0_norm = sqrt(e0_norm_sqr);
  Real e1_norm = sqrt(e1_norm_sqr);
  Real e2_norm = sqrt(e2_norm_sqr);
  Real e1_tilda_norm = sqrt(e1_tilda_norm_sqr);
  Real e2_tilda_norm = sqrt(e2_tilda_norm_sqr);

  Eigen::Vector3<Real> m0 = -(e1 - e1.dot(e0) * e0 / e0_norm_sqr).normalized();
  Eigen::Vector3<Real> m1 = (e2 - e2.dot(e1) * e1 / e1_norm_sqr).normalized();
  Eigen::Vector3<Real> m2 = (e1 - e1.dot(e2) * e2 / e2_norm_sqr).normalized();

  Eigen::Vector3<Real> m0_tilda = -(e1_tilda - e1_tilda.dot(e0) * e0 / e0_norm_sqr).normalized();
  Eigen::Vector3<Real> m1_tilda = (e2_tilda - e2_tilda.dot(e1_tilda) * e1_tilda / e1_tilda_norm_sqr).normalized();
  Eigen::Vector3<Real> m2_tilda = (e1_tilda - e1_tilda.dot(e2_tilda) * e2_tilda / e2_tilda_norm_sqr).normalized();

  Real h0 = -m0.dot(e1);
  Real h1 = m1.dot(e2);
  Real h2 = m2.dot(e1);

  Real h0_tilda = -m0_tilda.dot(e1_tilda);
  Real h1_tilda = m1_tilda.dot(e2_tilda);
  Real h2_tilda = m2_tilda.dot(e1_tilda);

  Eigen::Vector3<Real> n = e2.cross(e1).normalized();
  Eigen::Vector3<Real> n_tilda = e1_tilda.cross(e2_tilda).normalized();

  Real cos_a0 = e1.dot(e2) / (e1_norm * e2_norm);
  Real cos_a1 = e2.dot(e0) / (e2_norm * e0_norm);
  Real cos_a2 = -e0.dot(e1) / (e0_norm * e1_norm);

  Real cos_a0_tilda = e1_tilda.dot(e2_tilda) / (e1_tilda_norm * e2_tilda_norm);
  Real cos_a1_tilda = e2_tilda.dot(e0) / (e2_tilda_norm * e0_norm);
  Real cos_a2_tilda = -e0.dot(e1_tilda) / (e0_norm * e1_tilda_norm);
#define w00 (Real(1.0) / (h0 * h0))
#define w01 (Real(1.0) / (h0 * h1))
#define w02 (Real(1.0) / (h0 * h2))
#define w10 (Real(1.0) / (h1 * h0))
#define w11 (Real(1.0) / (h1 * h1))
#define w12 (Real(1.0) / (h1 * h2))
#define w20 (Real(1.0) / (h2 * h0))
#define w21 (Real(1.0) / (h2 * h1))
#define w22 (Real(1.0) / (h2 * h2))

#define w00_tilda (Real(1.0) / (h0_tilda * h0_tilda))
#define w01_tilda (Real(1.0) / (h0_tilda * h1_tilda))
#define w02_tilda (Real(1.0) / (h0_tilda * h2_tilda))
#define w10_tilda (Real(1.0) / (h1_tilda * h0_tilda))
#define w11_tilda (Real(1.0) / (h1_tilda * h1_tilda))
#define w12_tilda (Real(1.0) / (h1_tilda * h2_tilda))
#define w20_tilda (Real(1.0) / (h2_tilda * h0_tilda))
#define w21_tilda (Real(1.0) / (h2_tilda * h1_tilda))
#define w22_tilda (Real(1.0) / (h2_tilda * h2_tilda))

#define M0 (n * m0.transpose())
#define M1 (n * m1.transpose())
#define M2 (n * m2.transpose())
#define M0_tilda (n_tilda * m0_tilda.transpose())
#define M1_tilda (n_tilda * m1_tilda.transpose())
#define M2_tilda (n_tilda * m2_tilda.transpose())

#define N0 (M0 / e0_norm_sqr)
#define N0_tilda (M0_tilda / e0_norm_sqr)

#define P10 (w10 * cos_a2 * M0.transpose())
#define P11 (w11 * cos_a2 * M1.transpose())
#define P12 (w12 * cos_a2 * M2.transpose())
#define P20 (w20 * cos_a1 * M0.transpose())
#define P21 (w21 * cos_a1 * M1.transpose())
#define P22 (w22 * cos_a1 * M2.transpose())

#define P10_tilda (w10_tilda * cos_a2_tilda * M0_tilda.transpose())
#define P11_tilda (w11_tilda * cos_a2_tilda * M1_tilda.transpose())
#define P12_tilda (w12_tilda * cos_a2_tilda * M2_tilda.transpose())
#define P20_tilda (w20_tilda * cos_a1_tilda * M0_tilda.transpose())
#define P21_tilda (w21_tilda * cos_a1_tilda * M1_tilda.transpose())
#define P22_tilda (w22_tilda * cos_a1_tilda * M2_tilda.transpose())

#define Q0 (w00 * M0)
#define Q1 (w01 * M1)
#define Q2 (w02 * M2)
#define Q0_tilda (w00_tilda * M0_tilda)
#define Q1_tilda (w01_tilda * M1_tilda)
#define Q2_tilda (w02_tilda * M2_tilda)

#define S(A) (A + A.transpose())

  HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> H;
#define H_theta(i, j) H.m[0].block(i * 3, j * 3, 3, 3)
  H_theta(0, 0) = -S(Q0);
  H_theta(3, 3) = -S(Q0_tilda);
  H_theta(2, 2) = S(P22) - N0 + S(P22_tilda) - N0_tilda;
  H_theta(1, 1) = S(P11) - N0 + S(P11_tilda) - N0_tilda;
  H_theta(1, 0) = P10 - Q1;
  H_theta(2, 0) = P20 - Q2;
  H_theta(1, 3) = P10_tilda - Q1_tilda;
  H_theta(2, 3) = P20_tilda - Q2_tilda;
  H_theta(1, 2) = P12 + P21.transpose() + N0 + P12_tilda + P21_tilda.transpose() + N0_tilda;
  H_theta(0, 1) = H_theta(1, 0).transpose();
  H_theta(0, 2) = H_theta(2, 0).transpose();
  H_theta(3, 1) = H_theta(1, 3).transpose();
  H_theta(3, 2) = H_theta(2, 3).transpose();
  H_theta(2, 1) = H_theta(1, 2).transpose();

#if !defined(__CUDA_ARCH__) && false
  std::cout << "h0: " << h0 << std::endl;
  std::cout << "h1: " << h1 << std::endl;
  std::cout << "h2: " << h2 << std::endl;
  std::cout << "~h0: " << h0_tilda << std::endl;
  std::cout << "~h1: " << h1_tilda << std::endl;
  std::cout << "~h2: " << h2_tilda << std::endl;
  std::cout << "n: " << n.transpose() << std::endl;
  std::cout << "~n: " << n_tilda.transpose() << std::endl;
  std::cout << "m0: " << m0.transpose() << std::endl;
  std::cout << "m1: " << m1.transpose() << std::endl;
  std::cout << "m2: " << m2.transpose() << std::endl;
  std::cout << "~m0: " << m0_tilda.transpose() << std::endl;
  std::cout << "~m1: " << m1_tilda.transpose() << std::endl;
  std::cout << "~m2: " << m2_tilda.transpose() << std::endl;
  std::cout << "cos a0: " << cos_a0 << std::endl;
  std::cout << "cos a1: " << cos_a1 << std::endl;
  std::cout << "cos a2: " << cos_a2 << std::endl;
  std::cout << "cos ~a0: " << cos_a0_tilda << std::endl;
  std::cout << "cos ~a1: " << cos_a1_tilda << std::endl;
  std::cout << "cos ~a2: " << cos_a2_tilda << std::endl;
  std::cout << "M0: \n" << M0 << std::endl;
  std::cout << "M1: \n" << M1 << std::endl;
  std::cout << "M2: \n" << M2 << std::endl;
  std::cout << "~M0: \n" << M0_tilda << std::endl;
  std::cout << "~M1: \n" << M1_tilda << std::endl;
  std::cout << "~M2: \n" << M2_tilda << std::endl;
  std::cout << "N0: \n" << N0 << std::endl;
  std::cout << "N1: \n" << N1 << std::endl;
  std::cout << "N2: \n" << N2 << std::endl;
  std::cout << "~N0: \n" << N0_tilda << std::endl;
  std::cout << "~N1: \n" << N1_tilda << std::endl;
  std::cout << "~N2: \n" << N2_tilda << std::endl;
  std::cout << "P11: \n" << P11 << std::endl;
  std::cout << "P22: \n" << P22 << std::endl;
  std::cout << "~P11: \n" << P11_tilda << std::endl;
  std::cout << "~P22: \n" << P22_tilda << std::endl;
  std::cout << "S(P11): \n" << S(P11) << std::endl;
  std::cout << "S(P22): \n" << S(P22) << std::endl;
  std::cout << "S(~P11): \n" << S(P11_tilda) << std::endl;
  std::cout << "S(~P22): \n" << S(P22_tilda) << std::endl;
#endif
  return H;
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix3<Real> DihedralAngle<Real>::SubHessian(const InputType &V, int subdim) const {
  Eigen::Vector3<Real> e0 = V.col(2) - V.col(1);
  Eigen::Vector3<Real> e1 = V.col(0) - V.col(2);
  Eigen::Vector3<Real> e2 = V.col(0) - V.col(1);
  Eigen::Vector3<Real> e1_tilda = V.col(3) - V.col(2);
  Eigen::Vector3<Real> e2_tilda = V.col(3) - V.col(1);

  Real e0_norm_sqr = e0.squaredNorm();
  Real e1_norm_sqr = e1.squaredNorm();
  Real e2_norm_sqr = e2.squaredNorm();
  Real e1_tilda_norm_sqr = e1_tilda.squaredNorm();
  Real e2_tilda_norm_sqr = e2_tilda.squaredNorm();

  Real e0_norm = sqrt(e0_norm_sqr);
  Real e1_norm = sqrt(e1_norm_sqr);
  Real e2_norm = sqrt(e2_norm_sqr);
  Real e1_tilda_norm = sqrt(e1_tilda_norm_sqr);
  Real e2_tilda_norm = sqrt(e2_tilda_norm_sqr);

  Eigen::Vector3<Real> m0 = -(e1 - e1.dot(e0) * e0 / e0_norm_sqr).normalized();
  Eigen::Vector3<Real> m1 = (e2 - e2.dot(e1) * e1 / e1_norm_sqr).normalized();
  Eigen::Vector3<Real> m2 = (e1 - e1.dot(e2) * e2 / e2_norm_sqr).normalized();

  Eigen::Vector3<Real> m0_tilda = -(e1_tilda - e1_tilda.dot(e0) * e0 / e0_norm_sqr).normalized();
  Eigen::Vector3<Real> m1_tilda = (e2_tilda - e2_tilda.dot(e1_tilda) * e1_tilda / e1_tilda_norm_sqr).normalized();
  Eigen::Vector3<Real> m2_tilda = (e1_tilda - e1_tilda.dot(e2_tilda) * e2_tilda / e2_tilda_norm_sqr).normalized();

  Real h0 = -m0.dot(e1);
  Real h1 = m1.dot(e2);
  Real h2 = m2.dot(e1);

  Real h0_tilda = -m0_tilda.dot(e1_tilda);
  Real h1_tilda = m1_tilda.dot(e2_tilda);
  Real h2_tilda = m2_tilda.dot(e1_tilda);

  Eigen::Vector3<Real> n = e2.cross(e1).normalized();
  Eigen::Vector3<Real> n_tilda = e1_tilda.cross(e2_tilda).normalized();

  Real cos_a0 = e1.dot(e2) / (e1_norm * e2_norm);
  Real cos_a1 = e2.dot(e0) / (e2_norm * e0_norm);
  Real cos_a2 = -e0.dot(e1) / (e0_norm * e1_norm);

  Real cos_a0_tilda = e1_tilda.dot(e2_tilda) / (e1_tilda_norm * e2_tilda_norm);
  Real cos_a1_tilda = e2_tilda.dot(e0) / (e2_tilda_norm * e0_norm);
  Real cos_a2_tilda = -e0.dot(e1_tilda) / (e0_norm * e1_tilda_norm);

  switch (subdim) {
    case 0:
      return -S(Q0);
    case 1:
      return S(P11) - N0 + S(P11_tilda) - N0_tilda;
    case 2:
      return S(P22) - N0 + S(P22_tilda) - N0_tilda;
    case 3:
      return -S(Q0_tilda);
    default:
      return Eigen::Matrix3<Real>::Zero();
  }
}

template class DihedralAngle<float>;
template class DihedralAngle<double>;

// https://la.disneyresearch.com/wp-content/uploads/Discrete-Bending-Forces-and-Their-Jacobians-Paper.pdf
//
//   LM_DEVICE_FUNC HessianTensor<Real,
//                                OutputType::SizeAtCompileTime,
//                                InputType::SizeAtCompileTime>
//   Hessian(const InputType &V) const {
// #define e0 (V.col(2) - V.col(1))
// #define e1 (V.col(0) - V.col(2))
// #define e2 (V.col(0) - V.col(1))
// #define e1_tilda (V.col(3) - V.col(2))
// #define e2_tilda (V.col(3) - V.col(1))
//
// #define e0_norm_sqr (e0.squaredNorm())
// #define e1_norm_sqr (e1.squaredNorm())
// #define e2_norm_sqr (e2.squaredNorm())
// #define e1_tilda_norm_sqr (e1_tilda.squaredNorm())
// #define e2_tilda_norm_sqr (e2_tilda.squaredNorm())
//
// #define e0_norm (e0.norm())
// #define e1_norm (e1.norm())
// #define e2_norm (e2.norm())
// #define e1_tilda_norm (e1_tilda.norm())
// #define e2_tilda_norm (e2_tilda.norm())
//
// #define m0 (-(e1 - e1.dot(e0) * e0 / e0_norm_sqr).normalized())
// #define m1 ((e2 - e2.dot(e1) * e1 / e1_norm_sqr).normalized())
// #define m2 ((e1 - e1.dot(e2) * e2 / e2_norm_sqr).normalized())
//
// #define m0_tilda (-(e1_tilda - e1_tilda.dot(e0) * e0 /
// e0_norm_sqr).normalized()) #define m1_tilda ((e2_tilda -
// e2_tilda.dot(e1_tilda) * e1_tilda / e1_tilda_norm_sqr).normalized())
// #define m2_tilda ((e1_tilda - e1_tilda.dot(e2_tilda) * e2_tilda /
// e2_tilda_norm_sqr).normalized())
//
// #define h0 (-m0.dot(e1))
// #define h1 (m1.dot(e2))
// #define h2 (m2.dot(e1))
//
// #define h0_tilda (-m0_tilda.dot(e1_tilda))
// #define h1_tilda (m1_tilda.dot(e2_tilda))
// #define h2_tilda (m2_tilda.dot(e1_tilda))
//
// #define n (e2.cross(e1).normalized())
// #define n_tilda (e1_tilda.cross(e2_tilda).normalized())
//
// #define cos_a0 (e1.dot(e2) / (e1_norm * e2_norm))
// #define cos_a1 (e2.dot(e0) / (e2_norm * e0_norm))
// #define  cos_a2 (-e0.dot(e1) / (e0_norm * e1_norm))
//
// #define cos_a0_tilda (e1_tilda.dot(e2_tilda) / (e1_tilda_norm *
// e2_tilda_norm)) #define cos_a1_tilda (e2_tilda.dot(e0) / (e2_tilda_norm *
// e0_norm)) #define cos_a2_tilda (-e0.dot(e1_tilda) / (e0_norm *
// e1_tilda_norm))
//
// #define w00 (Real(1.0) / (h0 * h0))
// #define w01 (Real(1.0) / (h0 * h1))
// #define w02 (Real(1.0) / (h0 * h2))
// #define w10 (Real(1.0) / (h1 * h0))
// #define w11 (Real(1.0) / (h1 * h1))
// #define w12 (Real(1.0) / (h1 * h2))
// #define w20 (Real(1.0) / (h2 * h0))
// #define w21 (Real(1.0) / (h2 * h1))
// #define w22 (Real(1.0) / (h2 * h2))
//
// #define w00_tilda (Real(1.0) / (h0_tilda * h0_tilda))
// #define w01_tilda (Real(1.0) / (h0_tilda * h1_tilda))
// #define w02_tilda (Real(1.0) / (h0_tilda * h2_tilda))
// #define w10_tilda (Real(1.0) / (h1_tilda * h0_tilda))
// #define w11_tilda (Real(1.0) / (h1_tilda * h1_tilda))
// #define w12_tilda (Real(1.0) / (h1_tilda * h2_tilda))
// #define w20_tilda (Real(1.0) / (h2_tilda * h0_tilda))
// #define w21_tilda (Real(1.0) / (h2_tilda * h1_tilda))
// #define w22_tilda (Real(1.0) / (h2_tilda * h2_tilda))
//
// #define M0 (n * m0.transpose())
// #define M1 (n * m1.transpose())
// #define M2 (n * m2.transpose())
// #define M0_tilda (n_tilda * m0_tilda.transpose())
// #define M1_tilda (n_tilda * m1_tilda.transpose())
// #define M2_tilda (n_tilda * m2_tilda.transpose())
//
// #define N0 (M0 / e0_norm_sqr)
// #define N0_tilda (M0_tilda / e0_norm_sqr)
//
// #define P10 (w10 * cos_a2 * M0.transpose())
// #define P11 (w11 * cos_a2 * M1.transpose())
// #define P12 (w12 * cos_a2 * M2.transpose())
// #define P20 (w20 * cos_a1 * M0.transpose())
// #define P21 (w21 * cos_a1 * M1.transpose())
// #define P22 (w22 * cos_a1 * M2.transpose())
//
// #define P10_tilda (w10_tilda * cos_a2_tilda * M0_tilda.transpose())
// #define P11_tilda (w11_tilda * cos_a2_tilda * M1_tilda.transpose())
// #define P12_tilda (w12_tilda * cos_a2_tilda * M2_tilda.transpose())
// #define P20_tilda (w20_tilda * cos_a1_tilda * M0_tilda.transpose())
// #define P21_tilda (w21_tilda * cos_a1_tilda * M1_tilda.transpose())
// #define P22_tilda (w22_tilda * cos_a1_tilda * M2_tilda.transpose())
//
// #define Q0 (w00 * M0)
// #define Q1 (w01 * M1)
// #define Q2 (w02 * M2)
// #define Q0_tilda (w00_tilda * M0_tilda)
// #define Q1_tilda (w01_tilda * M1_tilda)
// #define Q2_tilda (w02_tilda * M2_tilda)
//
// #define S(A) (A + A.transpose())
//
//     HessianTensor<Real, OutputType::SizeAtCompileTime,
//                   InputType::SizeAtCompileTime>
//         H;
// #define H_theta(i, j) H.m[0].block(i * 3, j * 3, 3, 3)
//     H_theta(0, 0) = -S(Q0);
//     H_theta(3, 3) = -S(Q0_tilda);
//     H_theta(2, 2) = S(P22) - N0 + S(P22_tilda) - N0_tilda;
//     H_theta(1, 1) = S(P11) - N0 + S(P11_tilda) - N0_tilda;
//     H_theta(1, 0) = P10 - Q1;
//     H_theta(2, 0) = P20 - Q2;
//     H_theta(1, 3) = P10_tilda - Q1_tilda;
//     H_theta(2, 3) = P20_tilda - Q2_tilda;
//     H_theta(1, 2) = P12 + P21.transpose() + N0 + P12_tilda +
//                     P21_tilda.transpose() + N0_tilda;
//     H_theta(0, 1) = H_theta(1, 0).transpose();
//     H_theta(0, 2) = H_theta(2, 0).transpose();
//     H_theta(3, 1) = H_theta(1, 3).transpose();
//     H_theta(3, 2) = H_theta(2, 3).transpose();
//     H_theta(2, 1) = H_theta(1, 2).transpose();
//
// #if !defined(__CUDA_ARCH__) && false
//     std::cout << "h0: " << h0 << std::endl;
//     std::cout << "h1: " << h1 << std::endl;
//     std::cout << "h2: " << h2 << std::endl;
//     std::cout << "~h0: " << h0_tilda << std::endl;
//     std::cout << "~h1: " << h1_tilda << std::endl;
//     std::cout << "~h2: " << h2_tilda << std::endl;
//     std::cout << "n: " << n.transpose() << std::endl;
//     std::cout << "~n: " << n_tilda.transpose() << std::endl;
//     std::cout << "m0: " << m0.transpose() << std::endl;
//     std::cout << "m1: " << m1.transpose() << std::endl;
//     std::cout << "m2: " << m2.transpose() << std::endl;
//     std::cout << "~m0: " << m0_tilda.transpose() << std::endl;
//     std::cout << "~m1: " << m1_tilda.transpose() << std::endl;
//     std::cout << "~m2: " << m2_tilda.transpose() << std::endl;
//     std::cout << "cos a0: " << cos_a0 << std::endl;
//     std::cout << "cos a1: " << cos_a1 << std::endl;
//     std::cout << "cos a2: " << cos_a2 << std::endl;
//     std::cout << "cos ~a0: " << cos_a0_tilda << std::endl;
//     std::cout << "cos ~a1: " << cos_a1_tilda << std::endl;
//     std::cout << "cos ~a2: " << cos_a2_tilda << std::endl;
//     std::cout << "M0: \n" << M0 << std::endl;
//     std::cout << "M1: \n" << M1 << std::endl;
//     std::cout << "M2: \n" << M2 << std::endl;
//     std::cout << "~M0: \n" << M0_tilda << std::endl;
//     std::cout << "~M1: \n" << M1_tilda << std::endl;
//     std::cout << "~M2: \n" << M2_tilda << std::endl;
//     std::cout << "N0: \n" << N0 << std::endl;
//     std::cout << "N1: \n" << N1 << std::endl;
//     std::cout << "N2: \n" << N2 << std::endl;
//     std::cout << "~N0: \n" << N0_tilda << std::endl;
//     std::cout << "~N1: \n" << N1_tilda << std::endl;
//     std::cout << "~N2: \n" << N2_tilda << std::endl;
//     std::cout << "P11: \n" << P11 << std::endl;
//     std::cout << "P22: \n" << P22 << std::endl;
//     std::cout << "~P11: \n" << P11_tilda << std::endl;
//     std::cout << "~P22: \n" << P22_tilda << std::endl;
//     std::cout << "S(P11): \n" << S(P11) << std::endl;
//     std::cout << "S(P22): \n" << S(P22) << std::endl;
//     std::cout << "S(~P11): \n" << S(P11_tilda) << std::endl;
//     std::cout << "S(~P22): \n" << S(P22_tilda) << std::endl;
// #endif
//     return H;
//   }
//
//   LM_DEVICE_FUNC Eigen::Matrix3<Real> SubHessian(const InputType &V,
//                                                  int subdim) const {
//
// #define S(A) (A + A.transpose())
//     switch (subdim) {
//       case 0:
//         return -S(Q0);
//       case 1:
//         return S(P11) - N0 + S(P11_tilda) - N0_tilda;
//       case 2:
//         return S(P22) - N0 + S(P22_tilda) - N0_tilda;
//       case 3:
//         return -S(Q0_tilda);
//       default:
//         return Eigen::Matrix3<Real>::Zero();
//     }
//   }
// };
//
// #undef e0
// #undef e1
// #undef e2
// #undef e1_tilda
// #undef e2_tilda
//
// #undef e0_norm_sqr
// #undef e1_norm_sqr
// #undef e2_norm_sqr
// #undef e1_tilda_norm_sqr
// #undef e2_tilda_norm_sqr
//
// #undef e0_norm
// #undef e1_norm
// #undef e2_norm
// #undef e1_tilda_norm
// #undef e2_tilda_norm
//
// #undef m0
// #undef m1
// #undef m2
//
// #undef m0_tilda
// #undef m1_tilda
// #undef m2_tilda
//
// #undef h0
// #undef h1
// #undef h2
//
// #undef h0_tilda
// #undef h1_tilda
// #undef h2_tilda
//
// #undef n
// #undef n_tilda
//
//
// #undef cos_a0
// #undef cos_a1
// #undef cos_a2
//
// #undef cos_a0_tilda
// #undef cos_a1_tilda
// #undef cos_a2_tilda
//
// #undef w00
// #undef w01
// #undef w02
// #undef w10
// #undef w11
// #undef w12
// #undef w20
// #undef w21
// #undef w22
//
// #undef w00_tilda
// #undef w01_tilda
// #undef w02_tilda
// #undef w10_tilda
// #undef w11_tilda
// #undef w12_tilda
// #undef w20_tilda
// #undef w21_tilda
// #undef w22_tilda
//
// #undef M0
// #undef M1
// #undef M2
// #undef M0_tilda
// #undef M1_tilda
// #undef M2_tilda
//
// #undef N0
// #undef N0_tilda
//
// #undef P10
// #undef P11
// #undef P12
// #undef P20
// #undef P21
// #undef P22
//
// #undef P10_tilda
// #undef P11_tilda
// #undef P12_tilda
// #undef P20_tilda
// #undef P21_tilda
// #undef P22_tilda
//
// #undef Q0
// #undef Q1
// #undef Q2
// #undef Q0_tilda
// #undef Q1_tilda
// #undef Q2_tilda
//
// #undef S
}  // namespace grassland
