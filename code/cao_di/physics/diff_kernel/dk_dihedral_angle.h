#pragma once
#include "cao_di/physics/diff_kernel/dk_basics.h"

namespace CD {

template <typename Real>
struct DihedralAngleAssistEdgesToNormalsAxis {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 3> InputType;
  typedef Eigen::Matrix<Real, 3, 3> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &A) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &E) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, 9, 9> Jacobian(const InputType &E) const;

  LM_DEVICE_FUNC HessianTensor<Real, 9, 9> Hessian(const InputType &E) const;
};

template <typename Real>
struct DihedralAngleAssistNormalsAxisToSinCosTheta {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 3> InputType;
  typedef Eigen::Vector<Real, 2> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &normals_axis) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &N) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, 2, 9> Jacobian(const InputType &N) const;

  LM_DEVICE_FUNC HessianTensor<Real, 2, 9> Hessian(const InputType &A) const;
};

template <typename Real>
struct DihedralAngleAssistVerticesToEdges {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 4> InputType;
  typedef Eigen::Matrix<Real, 3, 3> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &V) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, 9, 12> Jacobian(const InputType &) const;

  LM_DEVICE_FUNC HessianTensor<Real, 9, 12> Hessian(const InputType &) const;
};

template <typename Real>
using DihedralAngleByEdges =
    Compose<Compose<DihedralAngleAssistEdgesToNormalsAxis<Real>, DihedralAngleAssistNormalsAxisToSinCosTheta<Real>>,
            Atan2<Real>>;

template <typename Real>
using DihedralAngleByVertices = Compose<DihedralAngleAssistVerticesToEdges<Real>, DihedralAngleByEdges<Real>>;

template <typename Real>
struct DihedralEnergy {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 4> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &V) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 12> Jacobian(const InputType &V) const;

  LM_DEVICE_FUNC HessianTensor<Real, 1, 12> Hessian(const InputType &V) const;

  Scalar rest_angle{0.0};
};

template <typename Real>
struct DihedralAngle {
  typedef Real Scalar;
  typedef Eigen::Matrix<Real, 3, 4> InputType;
  typedef Eigen::Matrix<Real, 1, 1> OutputType;

  LM_DEVICE_FUNC bool ValidInput(const InputType &) const;

  LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 12> Jacobian(const InputType &V) const;

  LM_DEVICE_FUNC OutputType operator()(const InputType &V) const;

  LM_DEVICE_FUNC HessianTensor<Real, 1, 12> Hessian(const InputType &V) const;

  LM_DEVICE_FUNC Eigen::Matrix3<Real> SubHessian(const InputType &V, int subdim) const;
};

LM_DEVICE_FUNC void DihedralAngleSubHessianJacobian0(const Matrix<float, 3, 4> &V,
                                                     Vector3<float> &jacobian,
                                                     Matrix3<float> &hessian);

LM_DEVICE_FUNC void DihedralAngleSubHessianJacobian1(const Matrix<float, 3, 4> &V,
                                                     Vector3<float> &jacobian,
                                                     Matrix3<float> &hessian);

LM_DEVICE_FUNC void DihedralAngleSubHessianJacobian2(const Matrix<float, 3, 4> &V,
                                                     Vector3<float> &jacobian,
                                                     Matrix3<float> &hessian);

LM_DEVICE_FUNC void DihedralAngleSubHessianJacobian3(const Matrix<float, 3, 4> &V,
                                                     Vector3<float> &jacobian,
                                                     Matrix3<float> &hessian);

LM_DEVICE_FUNC void DihedralAngleSubHessianJacobian(const Matrix<float, 3, 4> &V,
                                                    Vector3<float> &jacobian,
                                                    Matrix3<float> &hessian,
                                                    int subdim);

}  // namespace CD
