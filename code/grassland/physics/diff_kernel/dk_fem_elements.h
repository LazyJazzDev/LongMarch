#pragma once
#include "grassland/physics/diff_kernel/dk_basics.h"

namespace grassland {

template <typename Real>
struct FEMTetrahedronDeformationGradient {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 4> InputType;
  typedef Eigen::Matrix<Real, 3, 3> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &) const {
    return true;
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &V) const {
    OutputType Ds;
    Ds.col(0) = V.col(1) - V.col(0);
    Ds.col(1) = V.col(2) - V.col(0);
    Ds.col(2) = V.col(3) - V.col(0);
    return Ds * Dm.inverse();
  }

  LM_DEVICE_FUNC Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Jacobian(
      const InputType &) const {
    Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> J;
    J.setZero();
    for (int i = 0; i < 3; i++) {
      J(i, i) = -1;
      J(i, i + 3) = 1;
      J(i + 3, i) = -1;
      J(i + 3, i + 6) = 1;
      J(i + 6, i) = -1;
      J(i + 6, i + 9) = 1;
    }
    Eigen::Matrix3<Real> Dm_inv = Dm.inverse();
    Eigen::Matrix3<Real> Dm_inv_t = Dm_inv.transpose();

    Eigen::Matrix<Real, 9, 9> Dm_inv_t_jacobian;
    Dm_inv_t_jacobian.setZero();
    for (int i = 0; i < 3; i++) {
      Dm_inv_t_jacobian(i, i) = Dm_inv_t(0, 0);
      Dm_inv_t_jacobian(i, i + 3) = Dm_inv_t(0, 1);
      Dm_inv_t_jacobian(i, i + 6) = Dm_inv_t(0, 2);
      Dm_inv_t_jacobian(i + 3, i) = Dm_inv_t(1, 0);
      Dm_inv_t_jacobian(i + 3, i + 3) = Dm_inv_t(1, 1);
      Dm_inv_t_jacobian(i + 3, i + 6) = Dm_inv_t(1, 2);
      Dm_inv_t_jacobian(i + 6, i) = Dm_inv_t(2, 0);
      Dm_inv_t_jacobian(i + 6, i + 3) = Dm_inv_t(2, 1);
      Dm_inv_t_jacobian(i + 6, i + 6) = Dm_inv_t(2, 2);
    }
    // Dm_inv_t_jacobian.block<3, 3>(0, 0) =
    //     Eigen::Matrix<Real, 3, 3>::Identity() * Dm_inv_t(0, 0);
    // Dm_inv_t_jacobian.block<3, 3>(0, 3) =
    //     Eigen::Matrix<Real, 3, 3>::Identity() * Dm_inv_t(0, 1);
    // Dm_inv_t_jacobian.block<3, 3>(0, 6) =
    //     Eigen::Matrix<Real, 3, 3>::Identity() * Dm_inv_t(0, 2);
    // Dm_inv_t_jacobian.block<3, 3>(3, 0) =
    //     Eigen::Matrix<Real, 3, 3>::Identity() * Dm_inv_t(1, 0);
    // Dm_inv_t_jacobian.block<3, 3>(3, 3) =
    //     Eigen::Matrix<Real, 3, 3>::Identity() * Dm_inv_t(1, 1);
    // Dm_inv_t_jacobian.block<3, 3>(3, 6) =
    //     Eigen::Matrix<Real, 3, 3>::Identity() * Dm_inv_t(1, 2);
    // Dm_inv_t_jacobian.block<3, 3>(6, 0) =
    //     Eigen::Matrix<Real, 3, 3>::Identity() * Dm_inv_t(2, 0);
    // Dm_inv_t_jacobian.block<3, 3>(6, 3) =
    //     Eigen::Matrix<Real, 3, 3>::Identity() * Dm_inv_t(2, 1);
    // Dm_inv_t_jacobian.block<3, 3>(6, 6) =
    //     Eigen::Matrix<Real, 3, 3>::Identity() * Dm_inv_t(2, 2);
    return Dm_inv_t_jacobian * J;
  }

  LM_DEVICE_FUNC HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Hessian(
      const InputType &) const {
    HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> H;
    return H;
  }

  Eigen::Matrix3<Real> Dm;
};

template <typename Real>
struct FEMDeformationGradient3x2To3x3 {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 2> InputType;
  typedef Eigen::Matrix<Real, 3, 3> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &) const {
    return true;
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &F3x2) const {
    OutputType F;
    F.block(0, 0, 3, 2) = F3x2;
    F.col(2) = F3x2.col(0).cross(F3x2.col(1));
    return F;
  }

  LM_DEVICE_FUNC Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Jacobian(
      const InputType &F3x2) const {
    Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> J;
    Cross3<Real> cross3;
    J.setZero();
    for (int i = 0; i < 6; i++) {
      J(i, i) = 1;
    }
    J.block(6, 0, 3, 6) = cross3.Jacobian(F3x2);
    return J;
  }

  LM_DEVICE_FUNC HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Hessian(
      const InputType &F3x2) const {
    HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> H;
    auto H_cross3 = Cross3<Real>().Hessian(F3x2);
    H.m[6] = H_cross3.m[0];
    H.m[7] = H_cross3.m[1];
    H.m[8] = H_cross3.m[2];
    return H;
  }
};

template <typename Real>
struct FEMTriangleDeformationGradient3x2 {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 3> InputType;
  typedef Eigen::Matrix<Real, 3, 2> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &) const {
    return true;
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &V) const {
    OutputType Ds;
    Ds.col(0) = V.col(1) - V.col(0);
    Ds.col(1) = V.col(2) - V.col(0);
    return Ds * Dm.inverse();
  }

  LM_DEVICE_FUNC Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Jacobian(
      const InputType &) const {
    Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> J;
    J.setZero();
    for (int i = 0; i < 3; i++) {
      J(i, i) = -1;
      J(i, i + 3) = 1;
      J(i + 3, i) = -1;
      J(i + 3, i + 6) = 1;
    }
    // J.block<3, 3>(0, 0) = -Eigen::Matrix<Real, 3, 3>::Identity();
    // J.block<3, 3>(0, 3) = Eigen::Matrix<Real, 3, 3>::Identity();
    // J.block<3, 3>(3, 0) = -Eigen::Matrix<Real, 3, 3>::Identity();
    // J.block<3, 3>(3, 6) = Eigen::Matrix<Real, 3, 3>::Identity();
    Eigen::Matrix2<Real> Dm_inv = Dm.inverse();
    Eigen::Matrix2<Real> Dm_inv_t = Dm_inv.transpose();
    Eigen::Matrix<Real, 6, 6> Dm_inv_t_jacobian;
    Dm_inv_t_jacobian.setZero();
    for (int i = 0; i < 3; i++) {
      Dm_inv_t_jacobian(i, i) = Dm_inv_t(0, 0);
      Dm_inv_t_jacobian(i, i + 3) = Dm_inv_t(0, 1);
      Dm_inv_t_jacobian(i + 3, i) = Dm_inv_t(1, 0);
      Dm_inv_t_jacobian(i + 3, i + 3) = Dm_inv_t(1, 1);
    }
    // Dm_inv_t_jacobian.block<3, 3>(0, 0) =
    //     Eigen::Matrix<Real, 3, 3>::Identity() * Dm_inv_t(0, 0);
    // Dm_inv_t_jacobian.block<3, 3>(0, 3) =
    //     Eigen::Matrix<Real, 3, 3>::Identity() * Dm_inv_t(0, 1);
    // Dm_inv_t_jacobian.block<3, 3>(3, 0) =
    //     Eigen::Matrix<Real, 3, 3>::Identity() * Dm_inv_t(1, 0);
    // Dm_inv_t_jacobian.block<3, 3>(3, 3) =
    //     Eigen::Matrix<Real, 3, 3>::Identity() * Dm_inv_t(1, 1);
    return Dm_inv_t_jacobian * J;
  }

  LM_DEVICE_FUNC HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Hessian(
      const InputType &) const {
    HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> H;
    return H;
  }

  Eigen::Matrix2<Real> Dm;
};

template <typename Real>
struct FEMTriangleDeformationGradient3x3 {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 3> InputType;
  typedef Eigen::Matrix<Real, 3, 3> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &) const {
    return true;
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &V) const {
    FEMTriangleDeformationGradient3x2<Real> F3x2{Dm};
    FEMDeformationGradient3x2To3x3<Real> F3x2_to_F3x3;
    return F3x2_to_F3x3(F3x2(V));
  }

  LM_DEVICE_FUNC Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Jacobian(
      const InputType &V) const {
    FEMTriangleDeformationGradient3x2<Real> F3x2{Dm};
    FEMDeformationGradient3x2To3x3<Real> F3x2_to_F3x3;
    return F3x2_to_F3x3.Jacobian(F3x2(V)) * F3x2.Jacobian(V);
  }

  LM_DEVICE_FUNC HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Hessian(
      const InputType &V) const {
    FEMTriangleDeformationGradient3x2<Real> F3x2{Dm};
    FEMDeformationGradient3x2To3x3<Real> F3x2_to_F3x3;
    return F3x2_to_F3x3.Hessian(F3x2(V)) * F3x2.Jacobian(V);
  }

  Eigen::Matrix2<Real> Dm;
};

}  // namespace grassland
