#pragma once
#include "grassland/physics/basic_functions.h"

namespace grassland {
template <typename Real>
struct ElasticNeoHookean {
  ElasticNeoHookean(Real mu = 1.0, Real lambda = 1.0) : mu(mu), lambda(lambda) {
  }

  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 3> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &F) const {
    return F.determinant() > 0;
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &F) const {
    Determinant3<Real> det3;
    auto J = det3(F).value();
    auto log_J = log(J);
    Eigen::Map<const Eigen::Vector<Real, 9>> F_map(F.data());
    auto I2 = F_map.dot(F_map);
    OutputType res{0.5 * mu * (I2 - 3.0) - mu * log_J +
                   0.5 * lambda * log_J * log_J};
    return res;
  }

  LM_DEVICE_FUNC Eigen::
      Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime>
      Jacobian(const InputType &F) const {
    LogDeterminant3<Real> log_J;
    LogSquareDeterminant3<Real> log_2_J;
    Eigen::Map<const Eigen::RowVector<Real, 9>> F_map(F.data());
    return mu * (F_map - log_J.Jacobian(F)) +
           0.5 * lambda * log_2_J.Jacobian(F);
  }

  LM_DEVICE_FUNC HessianTensor<Real,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Hessian(const InputType &F) const {
    HessianTensor<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        H;
    LogDeterminant3<Real> log_J;
    LogSquareDeterminant3<Real> log_2_J;
    H.m[0] =
        mu * (Eigen::Matrix<Real, 9, 9>::Identity() - log_J.Hessian(F).m[0]) +
        0.5 * lambda * log_2_J.Hessian(F).m[0];
    return H;
  }

  Real mu;
  Real lambda;
};

}  // namespace grassland
