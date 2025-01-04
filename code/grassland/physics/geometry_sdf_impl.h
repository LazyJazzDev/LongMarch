#pragma once
#include "grassland/physics/geometry_sdf.h"

namespace grassland {
template <typename Real>
LM_DEVICE_FUNC bool SphereSDF<Real>::ValidInput(const InputType &v) const {
  return (v - center).norm() > algebra::Eps<Real>() * 100;
}

template <typename Real>
LM_DEVICE_FUNC typename SphereSDF<Real>::OutputType SphereSDF<Real>::operator()(const InputType &v) const {
  return OutputType{(v - center).norm() - radius};
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, SphereSDF<Real>::OutputType::SizeAtCompileTime, SphereSDF<Real>::InputType::SizeAtCompileTime> SphereSDF<Real>::Jacobian(const InputType &v) const {
  return (v - center).normalized().transpose();
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, SphereSDF<Real>::OutputType::SizeAtCompileTime, SphereSDF<Real>::InputType::SizeAtCompileTime> SphereSDF<Real>::Hessian(const InputType &v) const {
  HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> H;
  auto v_hat = (v - center).normalized().derived();
  H.m[0] = (H.m[0].Identity() - v_hat * v_hat.transpose()) / (v - center).norm();
  return H;
}

template class SphereSDF<float>;
template class SphereSDF<double>;

template <typename Real>
LM_DEVICE_FUNC bool LineSDF<Real>::ValidInput(const LineSDF<Real>::InputType &v) const {
  return operator()(v).value() > algebra::Eps<Real>() * 100;
}

template <typename Real>
LM_DEVICE_FUNC LineSDF<Real>::OutputType LineSDF<Real>::operator()(const LineSDF<Real>::InputType &v) const {
  Eigen::Vector3<Real> ab = B - A;
  Real t = (v - A).dot(ab) / ab.squaredNorm();
  t = max(Real(0), min(Real(1), t));
  return OutputType{(v - (A + t * ab)).norm()};
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, LineSDF<Real>::OutputType::SizeAtCompileTime, LineSDF<Real>::InputType::SizeAtCompileTime> LineSDF<Real>::Jacobian(const InputType &v) const {
  Eigen::Vector3<Real> ab = B - A;
  Real t = (v - A).dot(ab) / ab.squaredNorm();
  t = max(Real(0), min(Real(1), t));
  return (v - (A + t * ab)).normalized();
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, LineSDF<Real>::OutputType::SizeAtCompileTime, LineSDF<Real>::InputType::SizeAtCompileTime> LineSDF<Real>::Hessian(const InputType &v) const {
  HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> H;
  Eigen::Vector3<Real> ab = B - A;
  Real t = (v - A).dot(ab) / ab.squaredNorm();
  Eigen::Vector3<Real> center = A + max(Real(0), min(Real(1), t)) * ab;
  if (t < Real(0.0) || t > Real(1.0)) {
    auto v_hat = (v - center).normalized().derived();
    H.m[0] = (H.m[0].Identity() - v_hat * v_hat.transpose()) / (v - center).norm();
  } else {
    auto v_hat = (v - center).normalized().derived();
    auto ab_hat = ab.normalized().derived();
    H.m[0] = (H.m[0].Identity() - ab_hat * ab_hat.transpose() - v_hat * v_hat.transpose()) / (v - center).norm();
  }
  return H;
}

template class LineSDF<float>;
template class LineSDF<double>;

template <typename Real>
LM_DEVICE_FUNC bool CapsuleSDF<Real>::ValidInput(const InputType &v) const {
  return LineSDF<Real>{A, B}(v).value() > algebra::Eps<Real>() * 100;
}

template <typename Real>
LM_DEVICE_FUNC CapsuleSDF<Real>::OutputType CapsuleSDF<Real>::operator()(const InputType &v) const {
  Eigen::Vector3<Real> ab = B - A;
  Real t = (v - A).dot(ab) / ab.squaredNorm();
  t = max(Real(0), min(Real(1), t));
  return OutputType{(v - (A + t * ab)).norm() - radius};
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, CapsuleSDF<Real>::OutputType::SizeAtCompileTime, CapsuleSDF<Real>::InputType::SizeAtCompileTime> CapsuleSDF<Real>::Jacobian(const InputType &v) const {
  Eigen::Vector3<Real> ab = B - A;
  Real t = (v - A).dot(ab) / ab.squaredNorm();
  t = max(Real(0), min(Real(1), t));
  return (v - (A + t * ab)).normalized();
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, CapsuleSDF<Real>::OutputType::SizeAtCompileTime, CapsuleSDF<Real>::InputType::SizeAtCompileTime> CapsuleSDF<Real>::Hessian(const InputType &v) const {
  HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> H;
  Eigen::Vector3<Real> ab = B - A;
  Real t = (v - A).dot(ab) / ab.squaredNorm();
  Eigen::Vector3<Real> center = A + max(Real(0), min(Real(1), t)) * ab;
  if (t < Real(0.0) || t > Real(1.0)) {
    auto v_hat = (v - center).normalized().derived();
    H.m[0] = (H.m[0].Identity() - v_hat * v_hat.transpose()) / (v - center).norm();
  } else {
    auto v_hat = (v - center).normalized().derived();
    auto ab_hat = ab.normalized().derived();
    H.m[0] = (H.m[0].Identity() - ab_hat * ab_hat.transpose() - v_hat * v_hat.transpose()) / (v - center).norm();
  }
  return H;
}

template class CapsuleSDF<float>;
template class CapsuleSDF<double>;

template <typename Real>
LM_DEVICE_FUNC bool CubeSDF<Real>::ValidInput(const InputType &p) const {
  return true;
}

template <typename Real>
LM_DEVICE_FUNC CubeSDF<Real>::OutputType CubeSDF<Real>::operator()(const InputType &p) const {
  Eigen::Vector3<Real> p0 = p - center;
  Eigen::Vector3<Real> q = p0.cwiseAbs() - Eigen::Vector3<Real>(size, size, size);
  if (q.maxCoeff() > 0) {
    q = q.cwiseMax(Eigen::Vector3<Real>(0, 0, 0));
    return OutputType{q.norm()};
  } else {
    return OutputType{q.maxCoeff()};
  }
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, CubeSDF<Real>::OutputType::SizeAtCompileTime, CubeSDF<Real>::InputType::SizeAtCompileTime> CubeSDF<Real>::Jacobian(const InputType &p) const {
  Eigen::Vector3<Real> p0 = p - center;
  Eigen::Vector3<Real> q = p0.cwiseAbs() - Eigen::Vector3<Real>(size, size, size);
  if (q.maxCoeff() > 0) {
    q = q.cwiseMax(Eigen::Vector3<Real>(0, 0, 0));
    // multiply by sign to get the correct direction
    for (int i = 0; i < 3; i++) {
      if (p0(i) < 0) {
        q(i) = -q(i);
      }
    }
    return q.normalized().transpose();
  } else {
    // Find max component
    int max_idx = 0;
    for (int i = 1; i < 3; i++) {
      if (q(i) > q(max_idx)) {
        max_idx = i;
      }
    }
    Eigen::RowVector3<Real> result = Eigen::RowVector3<Real>::Zero();
    result(max_idx) = 1;
    if (p0(max_idx) < 0) {
      result(max_idx) = -1;
    }
    return result;
  }
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, CubeSDF<Real>::OutputType::SizeAtCompileTime, CubeSDF<Real>::InputType::SizeAtCompileTime> CubeSDF<Real>::Hessian(const InputType &v) const {
  HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> H;
  Eigen::Vector3<Real> v_diff = v - center;
  Eigen::Vector3<Real> p = v_diff;
  H.m[0] = Eigen::Matrix3<Real>::Identity();
  for (int dim = 0; dim < 3; dim++) {
    if (v(dim) < -size) {
      p(dim) = -size;
    } else if (v(dim) > size) {
      p(dim) = size;
    } else {
      p(dim) = v_diff(dim);
      H.m[0](dim, dim) = 0;
    }
  }
  v_diff = v_diff - p;
  Real v_diff_norm = v_diff.norm();
  if (v_diff_norm > algebra::Eps<Real>()) {
    v_diff = v_diff / v_diff_norm;
    H.m[0] = (H.m[0] - v_diff * v_diff.transpose()) / v_diff_norm;
    return H;
  } else {
    return {};
  }
}

template class CubeSDF<float>;
template class CubeSDF<double>;

}  // namespace grassland