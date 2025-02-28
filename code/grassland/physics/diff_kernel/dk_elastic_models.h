#pragma once
#include "grassland/physics/diff_kernel/dk_basics.h"
#include "grassland/physics/diff_kernel/dk_fem_elements.h"

namespace grassland {
template <typename Real>
struct ElasticNeoHookean {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 3> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &F) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &F) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 9> Jacobian(const InputType &F) const;

  LM_DEVICE_FUNC HessianTensor<Real, 1, 9> Hessian(const InputType &F) const;

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

  LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 9> Jacobian(const InputType &F) const;

  LM_DEVICE_FUNC HessianTensor<Real, 1, 9> Hessian(const InputType &F) const;

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

  LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 6> Jacobian(const InputType &F) const;

  LM_DEVICE_FUNC HessianTensor<Real, 1, 6> Hessian(const InputType &F) const;

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

  LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 6> Jacobian(const InputType &F) const;

  LM_DEVICE_FUNC HessianTensor<Real, 1, 6> Hessian(const InputType &F) const;

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

  LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 12> Jacobian(const InputType &V) const;

  LM_DEVICE_FUNC HessianTensor<Real, 1, 12> Hessian(const InputType &V) const;

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

  LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 12> Jacobian(const InputType &V) const;

  LM_DEVICE_FUNC HessianTensor<Real, 1, 12> Hessian(const InputType &V) const;

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

  LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 9> Jacobian(const InputType &V) const;

  LM_DEVICE_FUNC HessianTensor<Real, 1, 9> Hessian(const InputType &V) const;

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

  LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 9> Jacobian(const InputType &V) const;

  LM_DEVICE_FUNC HessianTensor<Real, 1, 9> Hessian(const InputType &V) const;

  Real mu{1.0};
  Real lambda{1.0};
  Eigen::Matrix2<Real> Dm;
};

}  // namespace grassland
