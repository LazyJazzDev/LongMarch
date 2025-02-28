#pragma once
#include "grassland/physics/diff_kernel/dk_basics.h"

namespace grassland {

template <typename Real>
struct PointSDF {
  typedef Real Scalar;
  typedef Eigen::Vector<Real, 3> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &v) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &v) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 3> Jacobian(const InputType &v) const;

  LM_DEVICE_FUNC HessianTensor<Real, 1, 3> Hessian(const InputType &v) const;

  Eigen::Vector3<Real> position{0, 0, 0};
};

template <typename Real>
struct SphereSDF {
  typedef Real Scalar;
  typedef Eigen::Vector<Real, 3> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &v) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &v) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 3> Jacobian(const InputType &v) const;

  LM_DEVICE_FUNC HessianTensor<Real, 1, 3> Hessian(const InputType &v) const;

  Eigen::Vector3<Real> center{0, 0, 0};
  Real radius{1.0};
};

template <typename Real>
struct SegmentSDF {
  typedef Real Scalar;
  typedef Eigen::Vector<Real, 3> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &v) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &v) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 3> Jacobian(const InputType &v) const;

  LM_DEVICE_FUNC HessianTensor<Real, 1, 3> Hessian(const InputType &v) const;

  Eigen::Vector3<Real> A{0, 0, 0};
  Eigen::Vector3<Real> B{1, 1, 1};
};

template <typename Real>
struct CapsuleSDF {
  typedef Real Scalar;
  typedef Eigen::Vector<Real, 3> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &v) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &v) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 3> Jacobian(const InputType &v) const;

  LM_DEVICE_FUNC HessianTensor<Real, 1, 3> Hessian(const InputType &v) const;

  Eigen::Vector3<Real> A{0, 0, 0};
  Eigen::Vector3<Real> B{1, 1, 1};
  Real radius{1.0};
};

template <typename Real>
struct CubeSDF {
  typedef Real Scalar;
  typedef Eigen::Vector<Real, 3> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &p) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &p) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 3> Jacobian(const InputType &p) const;

  LM_DEVICE_FUNC HessianTensor<Real, 1, 3> Hessian(const InputType &v) const;

  Eigen::Vector3<Real> center{0, 0, 0};
  Real size{1.0};
};

template <typename Real>
struct PlaneSDF {
  typedef Real Scalar;
  typedef Eigen::Vector<Real, 3> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &v) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &v) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 3> Jacobian(const InputType &v) const;

  LM_DEVICE_FUNC HessianTensor<Real, 1, 3> Hessian(const InputType &v) const;

  Eigen::Vector3<Real> normal{0, 1, 0};
  Real d{0.0};
};

template <typename Real>
struct LineSDF {
  typedef Real Scalar;
  typedef Eigen::Vector<Real, 3> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &v) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &v) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 3> Jacobian(const InputType &v) const;

  LM_DEVICE_FUNC HessianTensor<Real, 1, 3> Hessian(const InputType &v) const;

  Eigen::Vector3<Real> origin{0, 0, 0};
  Eigen::Vector3<Real> direction{0, 1, 0};
};

}  // namespace grassland
