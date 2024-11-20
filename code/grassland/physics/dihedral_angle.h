#pragma once
#include "grassland/physics/basic_functions.h"

namespace grassland {

template <typename Real>
struct DihedralAngleAssistEdgesToNormalsAxis {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 3> InputType;
  typedef Eigen::Matrix<Real, 3, 3> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &A) const {
    return A.col(0).cross(A.col(1)).norm() > algebra::Eps<Real>() * 100 &&
           A.col(1).cross(A.col(2)).norm() > algebra::Eps<Real>() * 100;
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &E) const {
    CrossNormalized<Real> cross_normalized;
    VecNormalized<Real> vec_normalized;
    OutputType normals_axis;
    normals_axis.col(0) = cross_normalized(E.block(0, 0, 3, 2));
    normals_axis.col(1) = -cross_normalized(E.block(0, 1, 3, 2));
    normals_axis.col(2) = vec_normalized(E.col(1));
    return normals_axis;
  }

  LM_DEVICE_FUNC Eigen::
      Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime>
      Jacobian(const InputType &E) const {
    Eigen::Matrix<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        J;
    J.setZero();
    CrossNormalized<Real> cross_normalized;
    VecNormalized<Real> vec_normalized;
    J.block(0, 0, 3, 6) = cross_normalized.Jacobian(E.block(0, 0, 3, 2));
    J.block(3, 3, 3, 6) = -cross_normalized.Jacobian(E.block(0, 1, 3, 2));
    J.block(6, 3, 3, 3) = vec_normalized.Jacobian(E.col(1));
    return J;
  }

  LM_DEVICE_FUNC HessianTensor<Real,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Hessian(const InputType &E) const {
    HessianTensor<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        H;
    CrossNormalized<Real> cross_normalized;
    VecNormalized<Real> vec_normalized;
    auto cross_normalized_hessian =
        cross_normalized.Hessian(E.block(0, 0, 3, 2));
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
};

template <typename Real>
struct DihedralAngleAssistNormalsAxisToSinCosTheta {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 3> InputType;
  typedef Eigen::Vector<Real, 2> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &normals_axis) const {
    return true;
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &N) const {
    Determinant3<Real> det3;
    Dot<Real> dot;
    return OutputType{det3(N).value(), dot(N.col(0), N.col(1)).value()};
  }

  LM_DEVICE_FUNC Eigen::
      Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime>
      Jacobian(const InputType &N) const {
    Eigen::Matrix<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        J;
    J.setZero();
    Determinant3<Real> det3;
    Dot<Real> dot;
    J.row(0) = det3.Jacobian(N);
    J.block(1, 0, 1, 6) = dot.Jacobian(N.block(0, 0, 3, 2));
    return J;
  }

  LM_DEVICE_FUNC HessianTensor<Real,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Hessian(const InputType &A) const {
    HessianTensor<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        H;
    H.m[0] = Determinant3<Real>().Hessian(A).m[0];
    H.m[1].block(0, 0, 6, 6) = Dot<Real>().Hessian(A.block(0, 0, 3, 2)).m[0];
    return H;
  }
};

template <typename Real>
struct DihedralAngleAssistVerticesToEdges {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 4> InputType;
  typedef Eigen::Matrix<Real, 3, 3> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &) const {
    return true;
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &V) const {
    OutputType res;
    res.col(0) = V.col(1) - V.col(0);
    res.col(1) = V.col(2) - V.col(1);
    res.col(2) = V.col(3) - V.col(2);
    return res;
  }

  LM_DEVICE_FUNC Eigen::
      Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime>
      Jacobian(const InputType &) const {
    Eigen::Matrix<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        J;
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

  LM_DEVICE_FUNC HessianTensor<Real,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Hessian(const InputType &) const {
    HessianTensor<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        H;
    return H;
  }
};

template <typename Real>
using DihedralAngleByEdges =
    Compose<Compose<DihedralAngleAssistEdgesToNormalsAxis<Real>,
                    DihedralAngleAssistNormalsAxisToSinCosTheta<Real>>,
            Atan2<Real>>;

template <typename Real>
using DihedralAngleByVertices =
    Compose<DihedralAngleAssistVerticesToEdges<Real>,
            DihedralAngleByEdges<Real>>;

template <typename Real>
struct DihedralEnergy {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 4> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &) const {
    return true;
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &V) const {
    DihedralAngleByVertices<Real> dihedral_angle;
    Real res = dihedral_angle(V).value();
    res -= rest_angle;
    return OutputType{res * res};
  }

  LM_DEVICE_FUNC Eigen::
      Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime>
      Jacobian(const InputType &V) const {
    DihedralAngleByVertices<Real> dihedral_angle;
    auto angle = dihedral_angle(V).value() - rest_angle;
    auto J = dihedral_angle.Jacobian(V);
    return (Real(2.0) * angle) * J;
  }

  LM_DEVICE_FUNC HessianTensor<Real,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Hessian(const InputType &V) const {
    DihedralAngleByVertices<Real> dihedral_angle;
    auto angle = dihedral_angle(V).value() - rest_angle;
    auto J = dihedral_angle.Jacobian(V);
    auto H = dihedral_angle.Hessian(V);
    H.m[0] = H.m[0] * angle + J.transpose() * J;
    return H * 2.0;
  }

  Scalar rest_angle{0.0};
};

}  // namespace grassland
