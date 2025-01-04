#include "grassland/physics/basic_functions.h"

namespace grassland {

template <typename Real>
struct SphereSDF {
  typedef Real Scalar;
  typedef Eigen::Vector<Real, 3> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &v) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &v) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Jacobian(const InputType &v) const;

  LM_DEVICE_FUNC HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Hessian(const InputType &v) const;

  Eigen::Vector3<Real> center{0, 0, 0};
  Real radius{1.0};
};

template <typename Real>
struct LineSDF {
  typedef Real Scalar;
  typedef Eigen::Vector<Real, 3> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &v) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &v) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Jacobian(const InputType &v) const;

  LM_DEVICE_FUNC HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Hessian(const InputType &v) const;

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

  LM_DEVICE_FUNC Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Jacobian(const InputType &v) const;

  LM_DEVICE_FUNC HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Hessian(const InputType &v) const;

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

  LM_DEVICE_FUNC Eigen::Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Jacobian(const InputType &p) const;

  LM_DEVICE_FUNC HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> Hessian(const InputType &v) const;

  Eigen::Vector3<Real> center{0, 0, 0};
  Real size{1.0};
};

}  // namespace grassland
