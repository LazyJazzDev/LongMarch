#include "grassland/physics/diff_kernel/dk_geometry_sdf.h"

#include <algorithm>

namespace grassland {

template <typename Real>
LM_DEVICE_FUNC bool PointSDF<Real>::ValidInput(const InputType &v) const {
  return (v - position).norm() > Eps<Real>() * 100;
}

template <typename Real>
LM_DEVICE_FUNC typename PointSDF<Real>::OutputType PointSDF<Real>::operator()(const InputType &v) const {
  return OutputType{(v - position).norm()};
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 3> PointSDF<Real>::Jacobian(const InputType &v) const {
  return (v - position).normalized().transpose();
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, 1, 3> PointSDF<Real>::Hessian(const InputType &v) const {
  HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> H;
  auto v_hat = (v - position).normalized().derived();
  H.m[0] = (H.m[0].Identity() - v_hat * v_hat.transpose()) / (v - position).norm();
  return H;
}

template class PointSDF<float>;
template class PointSDF<double>;

template <typename Real>
LM_DEVICE_FUNC bool SphereSDF<Real>::ValidInput(const InputType &v) const {
  return (v - center).norm() > Eps<Real>() * 100;
}

template <typename Real>
LM_DEVICE_FUNC typename SphereSDF<Real>::OutputType SphereSDF<Real>::operator()(const InputType &v) const {
  return OutputType{(v - center).norm() - radius};
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 3> SphereSDF<Real>::Jacobian(const InputType &v) const {
  return (v - center).normalized().transpose();
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, 1, 3> SphereSDF<Real>::Hessian(const InputType &v) const {
  HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> H;
  auto v_hat = (v - center).normalized().derived();
  H.m[0] = (H.m[0].Identity() - v_hat * v_hat.transpose()) / (v - center).norm();
  return H;
}

template class SphereSDF<float>;
template class SphereSDF<double>;

template <typename Real>
LM_DEVICE_FUNC bool SegmentSDF<Real>::ValidInput(const InputType &v) const {
  return operator()(v).value() > Eps<Real>() * 100;
}

template <typename Real>
LM_DEVICE_FUNC typename SegmentSDF<Real>::OutputType SegmentSDF<Real>::operator()(
    const SegmentSDF<Real>::InputType &v) const {
  Eigen::Vector3<Real> ab = B - A;
  Real t = (v - A).dot(ab) / ab.squaredNorm();
  t = fmax(Real(0), fmin(Real(1), t));
  return OutputType{(v - (A + t * ab)).norm()};
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 3> SegmentSDF<Real>::Jacobian(const InputType &v) const {
  Eigen::Vector3<Real> ab = B - A;
  Real t = (v - A).dot(ab) / ab.squaredNorm();
  t = fmax(Real(0), fmin(Real(1), t));
  return (v - (A + t * ab)).normalized();
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, 1, 3> SegmentSDF<Real>::Hessian(const InputType &v) const {
  HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> H;
  Eigen::Vector3<Real> ab = B - A;
  Real t = (v - A).dot(ab) / ab.squaredNorm();
  Eigen::Vector3<Real> center = A + fmax(Real(0), fmin(Real(1), t)) * ab;
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

template class SegmentSDF<float>;
template class SegmentSDF<double>;

template <typename Real>
LM_DEVICE_FUNC bool CapsuleSDF<Real>::ValidInput(const InputType &v) const {
  return SegmentSDF<Real>{A, B}(v).value() > Eps<Real>() * 100;
}

template <typename Real>
LM_DEVICE_FUNC typename CapsuleSDF<Real>::OutputType CapsuleSDF<Real>::operator()(const InputType &v) const {
  Eigen::Vector3<Real> ab = B - A;
  Real t = (v - A).dot(ab) / ab.squaredNorm();
  t = fmax(Real(0), fmin(Real(1), t));
  return OutputType{(v - (A + t * ab)).norm() - radius};
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 3> CapsuleSDF<Real>::Jacobian(const InputType &v) const {
  Eigen::Vector3<Real> ab = B - A;
  Real t = (v - A).dot(ab) / ab.squaredNorm();
  t = fmax(Real(0), fmin(Real(1), t));
  return (v - (A + t * ab)).normalized();
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, 1, 3> CapsuleSDF<Real>::Hessian(const InputType &v) const {
  HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> H;
  Eigen::Vector3<Real> ab = B - A;
  Real t = (v - A).dot(ab) / ab.squaredNorm();
  Eigen::Vector3<Real> center = A + fmax(Real(0), fmin(Real(1), t)) * ab;
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
LM_DEVICE_FUNC typename CubeSDF<Real>::OutputType CubeSDF<Real>::operator()(const InputType &p) const {
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
LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 3> CubeSDF<Real>::Jacobian(const InputType &p) const {
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
LM_DEVICE_FUNC HessianTensor<Real, 1, 3> CubeSDF<Real>::Hessian(const InputType &v) const {
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
  if (v_diff_norm > Eps<Real>()) {
    v_diff = v_diff / v_diff_norm;
    H.m[0] = (H.m[0] - v_diff * v_diff.transpose()) / v_diff_norm;
    return H;
  } else {
    return {};
  }
}

template class CubeSDF<float>;
template class CubeSDF<double>;

template <typename Real>
LM_DEVICE_FUNC bool PlaneSDF<Real>::ValidInput(const InputType &v) const {
  return true;
}

template <typename Real>
LM_DEVICE_FUNC typename PlaneSDF<Real>::OutputType PlaneSDF<Real>::operator()(const InputType &v) const {
  return OutputType{normal.dot(v) + d};
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 3> PlaneSDF<Real>::Jacobian(const InputType &v) const {
  return normal.transpose();
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, 1, 3> PlaneSDF<Real>::Hessian(const InputType &v) const {
  return HessianTensor<Real, Eigen::Matrix<Real, 1, 1>::SizeAtCompileTime,
                       Eigen::Matrix<Real, 3, 1>::SizeAtCompileTime>{};
}

template class PlaneSDF<float>;
template class PlaneSDF<double>;

template <typename Real>
LM_DEVICE_FUNC bool LineSDF<Real>::ValidInput(const InputType &v) const {
  return true;
}

template <typename Real>
LM_DEVICE_FUNC typename LineSDF<Real>::OutputType LineSDF<Real>::operator()(const InputType &v) const {
  return OutputType{(v - origin).cross(direction).norm()};
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 3> LineSDF<Real>::Jacobian(const InputType &v) const {
  Eigen::Vector3<Real> p = v - origin;
  return (p - p.dot(direction) * direction).normalized().transpose();
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, 1, 3> LineSDF<Real>::Hessian(const InputType &v) const {
  Eigen::Vector3<Real> t = (v - origin).cross(direction);
  HessianTensor<Real, Eigen::Matrix<Real, 1, 1>::SizeAtCompileTime, Eigen::Matrix<Real, 3, 1>::SizeAtCompileTime> H;
  Real t_norm = t.norm();
  t /= t_norm;
  H.m[0] = t * t.transpose() / t_norm;
  return H;
}

template class LineSDF<float>;
template class LineSDF<double>;

}  // namespace grassland
