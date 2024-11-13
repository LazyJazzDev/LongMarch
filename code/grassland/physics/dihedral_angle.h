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
    normals_axis.col(0) = cross_normalized(E.block<3, 2>(0, 0));
    normals_axis.col(1) = -cross_normalized(E.block<3, 2>(0, 1));
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
    J.block<3, 6>(0, 0) = cross_normalized.Jacobian(E.block<3, 2>(0, 0));
    J.block<3, 6>(3, 3) = -cross_normalized.Jacobian(E.block<3, 2>(0, 1));
    J.block<3, 3>(6, 3) = vec_normalized.Jacobian(E.col(1));
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
        cross_normalized.Hessian(E.block<3, 2>(0, 0));
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 6; j++) {
        for (int k = 0; k < 6; k++) {
          H(i, j, k) = cross_normalized_hessian(i, j, k);
        }
      }
    }
    cross_normalized_hessian = -cross_normalized.Hessian(E.block<3, 2>(0, 1));
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
    J.block<1, 6>(1, 0) = dot.Jacobian(N.block<3, 2>(0, 0));
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
    H.m[1].block<6, 6>(0, 0) = Dot<Real>().Hessian(A.block<3, 2>(0, 0)).m[0];
    return H;
  }
};

template <typename Real>
using DihedralAngleByEdges =
    Compose<Compose<DihedralAngleAssistEdgesToNormalsAxis<Real>,
                    DihedralAngleAssistNormalsAxisToSinCosTheta<Real>>,
            Atan2<Real>>;

}  // namespace grassland