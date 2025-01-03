#pragma once
#include "grassland/physics/basic_functions.h"
#include "grassland/physics/fem_elements.h"

namespace grassland {
template <typename Real>
struct ElasticNeoHookean {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 3> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &F) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &F) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Jacobian(const InputType &F) const;

  LM_DEVICE_FUNC HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Hessian(const InputType &F) const;

  Real mu{1.0};
  Real lambda{1.0};
};

template <typename Real>
struct ElasticNeoHookeanSimple {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 3> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &F) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &F) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Jacobian(const InputType &F) const;

  LM_DEVICE_FUNC HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Hessian(const InputType &F) const;

  Real mu{1.0};
  Real lambda{1.0};
};

template <typename Real>
struct ElasticNeoHookeanF3x2 {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 2> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &F) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &F) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Jacobian(const InputType &F) const;

  LM_DEVICE_FUNC HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Hessian(const InputType &F) const;

  Real mu{1.0};
  Real lambda{1.0};
};

template <typename Real>
struct ElasticNeoHookeanSimpleF3x2 {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 2> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &F) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &F) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Jacobian(const InputType &F) const;

  LM_DEVICE_FUNC HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Hessian(const InputType &F) const;

  Real mu{1.0};
  Real lambda{1.0};
};

template <typename Real>
struct ElasticNeoHookeanTetrahedron {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 4> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &V) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &V) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Jacobian(const InputType &V) const;

  LM_DEVICE_FUNC HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Hessian(const InputType &V) const;

  LM_DEVICE_FUNC Eigen::Matrix3<Real> SubHessian(const InputType &V, int dim) const;

  Real mu{1.0};
  Real lambda{1.0};
  Eigen::Matrix3<Real> Dm;
};

template <typename Real>
struct ElasticNeoHookeanSimpleTetrahedron {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 4> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &V) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &V) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Jacobian(const InputType &V) const;

  LM_DEVICE_FUNC HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Hessian(const InputType &V) const;

  LM_DEVICE_FUNC Eigen::Matrix3<Real> SubHessian(const InputType &V, int dim) const;

  Real mu{1.0};
  Real lambda{1.0};
  Eigen::Matrix3<Real> Dm;
};

template <typename Real>
struct ElasticNeoHookeanTriangle {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 3> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &V) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &V) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Jacobian(const InputType &V) const;

  LM_DEVICE_FUNC HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Hessian(const InputType &V) const;

  Real mu{1.0};
  Real lambda{1.0};
  Eigen::Matrix2<Real> Dm;
};

template <typename Real>
struct ElasticNeoHookeanSimpleTriangle {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 3> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &V) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &V) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Jacobian(const InputType &V) const;

  LM_DEVICE_FUNC HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Hessian(const InputType &V) const;

  Real mu{1.0};
  Real lambda{1.0};
  Eigen::Matrix2<Real> Dm;
};

}  // namespace grassland
