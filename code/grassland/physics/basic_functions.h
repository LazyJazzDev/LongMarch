#pragma once
#include "grassland/algebra/algebra.h"
#include "grassland/physics/hessian_tensor.h"
#include "grassland/physics/physics_util.h"

namespace grassland {

template <typename Func1, typename Func2>
struct Compose {
  Func1 f1;
  Func2 f2;

  typedef typename Func1::Scalar Scalar;
  typedef typename Func1::InputType InputType;
  typedef typename Func2::OutputType OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &x) const {
    return f1.ValidInput(x) && f2.ValidInput(f1(x));
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &x) const {
    return f2(f1(x));
  }

  LM_DEVICE_FUNC Eigen::Matrix<Scalar,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Jacobian(const InputType &x) const {
    return f2.Jacobian(f1(x)) * f1.Jacobian(x);
  }

  LM_DEVICE_FUNC HessianTensor<Scalar,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Hessian(const InputType &x) const {
    auto J1 = f1.Jacobian(x);
    auto J2 = f2.Jacobian(f1(x));
    auto H1 = f1.Hessian(x);
    auto H2 = f2.Hessian(f1(x));
    return H2 * J1 + J2 * H1;
  }
};

template <typename Real>
struct Determinant3 {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 3> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &A) const {
    return true;
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &A) const {
    OutputType result{A.determinant()};
    return result;
  }

  LM_DEVICE_FUNC Eigen::
      Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime>
      Jacobian(const InputType &A) const {
    Eigen::Matrix<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        J;
    J.block<1, 3>(0, 0) = A.col(1).cross(A.col(2)).transpose();
    J.block<1, 3>(0, 3) = A.col(2).cross(A.col(0)).transpose();
    J.block<1, 3>(0, 6) = A.col(0).cross(A.col(1)).transpose();
    return J;
  }

  LM_DEVICE_FUNC HessianTensor<Real,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Hessian(const InputType &A) const {
    HessianTensor<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        H;
    H.m[0].block<3, 3>(0, 3) = Skew3<Real>(-A.col(2));
    H.m[0].block<3, 3>(0, 6) = Skew3<Real>(A.col(1));
    H.m[0].block<3, 3>(3, 0) = Skew3<Real>(A.col(2));
    H.m[0].block<3, 3>(3, 6) = Skew3<Real>(-A.col(0));
    H.m[0].block<3, 3>(6, 0) = Skew3<Real>(-A.col(1));
    H.m[0].block<3, 3>(6, 3) = Skew3<Real>(A.col(0));
    return H;
  }
};

template <typename Real>
struct LogDeterminant3 {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 3> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &A) const {
    return A.determinant() > 0;
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &A) const {
    OutputType result{log(A.determinant())};
    return result;
  }

  LM_DEVICE_FUNC Eigen::
      Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime>
      Jacobian(const InputType &A) const {
    Determinant3<Real> det;
    return (1 / A.determinant()) * det.Jacobian(A);
  }

  LM_DEVICE_FUNC HessianTensor<Real,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Hessian(const InputType &A) const {
    HessianTensor<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        H;
    Determinant3<Real> det;
    auto J = det(A).value();
    auto det_jacobi = det.Jacobian(A);
    H.m[0] = (-1.0 / (J * J)) * det_jacobi.transpose() * det_jacobi +
             (1.0 / J) * det.Hessian(A).m[0];
    return H;
  }
};

template <typename Real>
struct LogSquareDeterminant3 {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 3> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &A) const {
    return A.determinant() > 0;
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &A) const {
    Real log_det = log(A.determinant());
    OutputType result{log_det * log_det};
    return result;
  }

  LM_DEVICE_FUNC Eigen::
      Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime>
      Jacobian(const InputType &A) const {
    Determinant3<Real> det;
    auto J = det(A).value();
    return 2.0 * (1.0 / J) * log(J) * det.Jacobian(A);
  }

  LM_DEVICE_FUNC HessianTensor<Real,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Hessian(const InputType &A) const {
    HessianTensor<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        H;
    Determinant3<Real> det;
    auto J = det(A).value();
    auto det_jacobi = det.Jacobian(A);
    H.m[0] =
        (2.0 / (J * J) * (1.0 - log(J))) * det_jacobi.transpose() * det_jacobi +
        (2.0 / J * log(J)) * det.Hessian(A).m[0];
    return H;
  }
};

template <typename Real, int dim = 3>
struct VecLength {
  typedef Real Scalar;
  typedef Eigen::Vector<Real, dim> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &v) const {
    return v.norm() > algebra::Eps<Real>() * 100;
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &v) const {
    return OutputType{v.norm()};
  }

  LM_DEVICE_FUNC Eigen::
      Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime>
      Jacobian(const InputType &v) const {
    return v.normalized().transpose();
  }

  LM_DEVICE_FUNC HessianTensor<Real,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Hessian(const InputType &v) const {
    HessianTensor<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        H;
    auto v_hat = v.normalized().derived();
    H.m[0] = (H.m[0].Identity() - v_hat * v_hat.transpose()) / v.norm();
    return H;
  }
};

template <typename Real, int dim = 3>
struct VecNormalized {
  typedef Real Scalar;
  typedef Eigen::Vector<Real, dim> InputType;
  typedef Eigen::Vector<Real, dim> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &v) const {
    return v.norm() > algebra::Eps<Real>() * 100;
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &v) const {
    return v.normalized();
  }

  LM_DEVICE_FUNC Eigen::
      Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime>
      Jacobian(const InputType &v) const {
    auto v_hat = v.normalized().derived();
    return (Eigen::Matrix<Real, dim, dim>::Identity() -
            v_hat * v_hat.transpose()) /
           v.norm();
  }

  LM_DEVICE_FUNC HessianTensor<Real,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Hessian(const InputType &v) const {
    HessianTensor<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        H;
    InputType v_hat = v.normalized();
    Real inv_v_norm_2 = v.norm();
    inv_v_norm_2 = 1.0 / (inv_v_norm_2 * inv_v_norm_2);

    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        for (int k = 0; k < dim; k++) {
          if (j == k) {
            H(i, j, k) -= v_hat[i] * inv_v_norm_2;
          }
          if (i == k) {
            H(i, j, k) -= v_hat[j] * inv_v_norm_2;
          }
          if (i == j) {
            H(i, j, k) -= v_hat[k] * inv_v_norm_2;
          }
          H(i, j, k) += 3.0 * v_hat[i] * v_hat[j] * v_hat[k] * inv_v_norm_2;
        }
      }
    }
    return H;
  }
};

template <typename Real>
struct Cross3 {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 2> InputType;
  typedef Eigen::Vector<Real, 3> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &v) const {
    return true;
  }

  LM_DEVICE_FUNC OutputType operator()(const Eigen::Vector3<Real> &u,
                                       const Eigen::Vector3<Real> &v) const {
    return u.cross(v);
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &v) const {
    return operator()(v.col(0), v.col(1));
  }

  LM_DEVICE_FUNC Eigen::
      Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime>
      Jacobian(const InputType &v) const {
    Eigen::Matrix<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        J;
    J.block<3, 3>(0, 0) = -Skew3<Real>(v.col(1));
    J.block<3, 3>(0, 3) = Skew3<Real>(v.col(0));
    return J;
  }

  LM_DEVICE_FUNC HessianTensor<Real,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Hessian(const InputType &v) const {
    HessianTensor<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        H;
    H(0, 5, 1) = 1.0;
    H(0, 1, 5) = 1.0;
    H(0, 4, 2) = -1.0;
    H(0, 2, 4) = -1.0;
    H(1, 5, 0) = -1.0;
    H(1, 0, 5) = -1.0;
    H(1, 3, 2) = 1.0;
    H(1, 2, 3) = 1.0;
    H(2, 4, 0) = 1.0;
    H(2, 0, 4) = 1.0;
    H(2, 3, 1) = -1.0;
    H(2, 1, 3) = -1.0;
    return H;
  }
};

template <typename Real>
struct Dot {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 2> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &v) const {
    return true;
  }

  LM_DEVICE_FUNC OutputType operator()(const Eigen::Vector3<Real> &u,
                                       const Eigen::Vector3<Real> &v) const {
    return OutputType{u.dot(v)};
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &v) const {
    return OutputType{operator()(v.col(0), v.col(1))};
  }

  LM_DEVICE_FUNC Eigen::
      Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime>
      Jacobian(const InputType &v) const {
    Eigen::Matrix<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        J;
    J.block<1, 3>(0, 0) = v.col(1).transpose();
    J.block<1, 3>(0, 3) = v.col(0).transpose();
    return J;
  }

  LM_DEVICE_FUNC HessianTensor<Real,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Hessian(const InputType &v) const {
    HessianTensor<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        H;
    H(0, 3, 0) = 1.0;
    H(0, 4, 1) = 1.0;
    H(0, 5, 2) = 1.0;
    H(0, 0, 3) = 1.0;
    H(0, 1, 4) = 1.0;
    H(0, 2, 5) = 1.0;
    return H;
  }
};

template <typename Real>
struct Atan2 {
  typedef Real Scalar;
  typedef Eigen::Vector<Real, 2> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &v) const {
    return v.norm() > algebra::Eps<Real>() * 100;
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &v) const {
    return OutputType{atan2(v[0], v[1])};
  }

  LM_DEVICE_FUNC Eigen::
      Matrix<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime>
      Jacobian(const InputType &v) const {
    Eigen::Matrix<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        J;
    auto len2 = v.squaredNorm();
    J[0] = v[1] / len2;
    J[1] = -v[0] / len2;
    return J;
  }

  LM_DEVICE_FUNC HessianTensor<Real,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Hessian(const InputType &v) const {
    HessianTensor<Real, OutputType::SizeAtCompileTime,
                  InputType::SizeAtCompileTime>
        H;
    auto len2 = v.squaredNorm();
    H(0, 0, 0) = -2.0 * v[0] * v[1] / (len2 * len2);
    H(0, 0, 1) = H(0, 1, 0) = (v[0] * v[0] - v[1] * v[1]) / (len2 * len2);
    H(0, 1, 1) = 2.0 * v[0] * v[1] / (len2 * len2);
    return H;
  }
};

template <typename Func>
struct MultiplyConstant {
  typedef typename Func::Scalar Scalar;
  typedef typename Func::InputType InputType;
  typedef typename Func::OutputType OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &v) const {
    return f.ValidInput(v);
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &v) const {
    return f(v) * s;
  }

  LM_DEVICE_FUNC Eigen::Matrix<Scalar,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Jacobian(const InputType &v) const {
    return f.Jacobian(v) * s;
  }

  LM_DEVICE_FUNC HessianTensor<Scalar,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Hessian(const InputType &v) const {
    return f.Hessian(v) * s;
  }

  Func f{};
  Scalar s{1.0};
};

template <typename Func>
struct PlusConstant {
  typedef typename Func::Scalar Scalar;
  typedef typename Func::InputType InputType;
  typedef typename Func::OutputType OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &v) const {
    return f.ValidInput(v);
  }

  LM_DEVICE_FUNC OutputType operator()(const InputType &v) const {
    return f(v) + s;
  }

  LM_DEVICE_FUNC Eigen::Matrix<Scalar,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Jacobian(const InputType &v) const {
    return f.Jacobian(v);
  }

  LM_DEVICE_FUNC HessianTensor<Scalar,
                               OutputType::SizeAtCompileTime,
                               InputType::SizeAtCompileTime>
  Hessian(const InputType &v) const {
    return f.Hessian(v);
  }

  Func f{};
  OutputType s{};
};

template <typename Real>
using CrossNormalized = Compose<Cross3<Real>, VecNormalized<Real>>;

}  // namespace grassland
