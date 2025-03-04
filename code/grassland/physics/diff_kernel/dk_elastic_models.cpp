#include "grassland/physics/diff_kernel/dk_elastic_models.h"

namespace grassland {
template <typename Real>
LM_DEVICE_FUNC bool ElasticNeoHookean<Real>::ValidInput(const InputType &F) const {
  return F.determinant() > 0;
}

template <typename Real>
LM_DEVICE_FUNC typename ElasticNeoHookean<Real>::OutputType ElasticNeoHookean<Real>::operator()(
    const InputType &F) const {
  Determinant3<Real> det3;
  auto J = det3(F).value();
  auto log_J = log(J);
  Eigen::Map<const Eigen::Vector<Real, 9>> F_map(F.data());
  auto I2 = F_map.dot(F_map);
  OutputType res{0.5 * mu * (I2 - 3.0) - mu * log_J + 0.5 * lambda * log_J * log_J};
  return res;
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 9> ElasticNeoHookean<Real>::Jacobian(const InputType &F) const {
  LogDeterminant3<Real> log_J;
  LogSquareDeterminant3<Real> log_2_J;
  Eigen::Map<const Eigen::RowVector<Real, 9>> F_map(F.data());
  return mu * (F_map - log_J.Jacobian(F)) + 0.5 * lambda * log_2_J.Jacobian(F);
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, 1, 9> ElasticNeoHookean<Real>::Hessian(const InputType &F) const {
  HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> H;
  LogDeterminant3<Real> log_J;
  LogSquareDeterminant3<Real> log_2_J;
  H.m[0] =
      mu * (Eigen::Matrix<Real, 9, 9>::Identity() - log_J.Hessian(F).m[0]) + 0.5 * lambda * log_2_J.Hessian(F).m[0];
  return H;
}

template class ElasticNeoHookean<float>;
template class ElasticNeoHookean<double>;

template <typename Real>
LM_DEVICE_FUNC bool ElasticNeoHookeanSimple<Real>::ValidInput(const InputType &F) const {
  return true;
}

template <typename Real>
LM_DEVICE_FUNC typename ElasticNeoHookeanSimple<Real>::OutputType ElasticNeoHookeanSimple<Real>::operator()(
    const InputType &F) const {
  Determinant3<Real> det3;
  auto J = det3(F).value();
  Eigen::Map<const Eigen::Vector<Real, 9>> F_map(F.data());
  auto I2 = F_map.dot(F_map);
  auto a = 1.0 + mu / lambda;
  OutputType res{0.5 * mu * (I2 - 3.0) + 0.5 * lambda * (J - a) * (J - a)};
  return res;
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 9> ElasticNeoHookeanSimple<Real>::Jacobian(const InputType &F) const {
  Determinant3<Real> det3;
  auto J = det3(F).value();
  Eigen::Map<const Eigen::Vector<Real, 9>> F_map(F.data());
  auto I2 = F_map.dot(F_map);
  auto a = 1.0 + mu / lambda;
  return mu * F_map.transpose() + lambda * (J - a) * det3.Jacobian(F);
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, 1, 9> ElasticNeoHookeanSimple<Real>::Hessian(const InputType &F) const {
  HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> H;

  Determinant3<Real> det3;
  auto J = det3(F).value();
  Eigen::Map<const Eigen::Vector<Real, 9>> F_map(F.data());
  auto I2 = F_map.dot(F_map);
  auto a = 1.0 + mu / lambda;
  auto det3_jacobi = det3.Jacobian(F);

  H.m[0] = mu * (Eigen::Matrix<Real, 9, 9>::Identity()) + lambda * (J - a) * det3.Hessian(F).m[0] +
           lambda * det3_jacobi.transpose() * det3_jacobi;

  return H;
}

template class ElasticNeoHookeanSimple<float>;
template class ElasticNeoHookeanSimple<double>;

template <typename Real>
LM_DEVICE_FUNC bool ElasticNeoHookeanF3x2<Real>::ValidInput(const InputType &F) const {
  return F.col(0).cross(F.col(1)).norm() > Eps<Real>() * 100;
}

template <typename Real>
LM_DEVICE_FUNC typename ElasticNeoHookeanF3x2<Real>::OutputType ElasticNeoHookeanF3x2<Real>::operator()(
    const InputType &F) const {
  CrossNorm<Real> cross_norm;
  auto J = cross_norm(F).value();
  auto log_J = log(J);
  Eigen::Map<const Eigen::Vector<Real, 6>> F_map(F.data());
  auto I2 = F_map.dot(F_map) + Real(1.0);
  OutputType res{0.5 * mu * (I2 - 3.0) - mu * log_J + 0.5 * lambda * log_J * log_J};
  return res;
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 6> ElasticNeoHookeanF3x2<Real>::Jacobian(const InputType &F) const {
  CrossNorm<Real> cross_norm;
  auto J = cross_norm(F).value();
  auto log_J = log(J);
  Eigen::Map<const Eigen::Vector<Real, 6>> F_map(F.data());
  auto inv_J = 1.0 / J;
  auto cross_norm_J = cross_norm.Jacobian(F);
  return mu * (F_map.transpose() - inv_J * cross_norm_J) + 0.5 * lambda * (2.0 * log_J * inv_J * cross_norm_J);
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, 1, 6> ElasticNeoHookeanF3x2<Real>::Hessian(const InputType &F) const {
  HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> H;
  CrossNorm<Real> cross_norm;
  auto J = cross_norm(F).value();
  auto log_J = log(J);
  Eigen::Map<const Eigen::Vector<Real, 6>> F_map(F.data());
  auto inv_J = 1.0 / J;
  auto inv_J2 = inv_J * inv_J;
  auto cross_norm_J = cross_norm.Jacobian(F);
  auto cross_norm_H = cross_norm.Hessian(F);
  H.m[0] = mu * (Eigen::Matrix<Real, 6, 6>::Identity() -
                 (inv_J * cross_norm_H.m[0] - inv_J2 * cross_norm_J.transpose() * cross_norm_J)) +
           0.5 * lambda *
               (2.0 * log_J * inv_J * cross_norm_H.m[0] +
                (2.0 * inv_J2 * (1.0 - log_J)) * cross_norm_J.transpose() * cross_norm_J);
  return H;
}

template class ElasticNeoHookeanF3x2<float>;
template class ElasticNeoHookeanF3x2<double>;

template <typename Real>
LM_DEVICE_FUNC bool ElasticNeoHookeanSimpleF3x2<Real>::ValidInput(const InputType &F) const {
  return F.col(0).cross(F.col(1)).norm() > Eps<Real>() * 100;
}

template <typename Real>
LM_DEVICE_FUNC typename ElasticNeoHookeanSimpleF3x2<Real>::OutputType ElasticNeoHookeanSimpleF3x2<Real>::operator()(
    const InputType &F) const {
  CrossNorm<Real> cross_norm;
  auto J = cross_norm(F).value();
  auto log_J = log(J);
  Eigen::Map<const Eigen::Vector<Real, 6>> F_map(F.data());
  auto I2 = F_map.dot(F_map) + Real(1.0);
  auto a = 1.0 + mu / lambda;
  OutputType res{0.5 * mu * (I2 - 3.0) + 0.5 * lambda * (J - a) * (J - a)};
  return res;
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 6> ElasticNeoHookeanSimpleF3x2<Real>::Jacobian(const InputType &F) const {
  CrossNorm<Real> cross_norm;
  auto J = cross_norm(F).value();
  Eigen::Map<const Eigen::Vector<Real, 6>> F_map(F.data());
  auto cross_norm_J = cross_norm.Jacobian(F);
  auto a = 1.0 + mu / lambda;
  return mu * F_map.transpose() + lambda * (J - a) * cross_norm_J;
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, 1, 6> ElasticNeoHookeanSimpleF3x2<Real>::Hessian(const InputType &F) const {
  HessianTensor<Real, OutputType::SizeAtCompileTime, InputType::SizeAtCompileTime> H;
  CrossNorm<Real> cross_norm;
  auto J = cross_norm(F).value();
  auto cross_norm_J = cross_norm.Jacobian(F);
  auto cross_norm_H = cross_norm.Hessian(F);
  auto a = 1.0 + mu / lambda;
  H.m[0] = mu * Eigen::Matrix<Real, 6, 6>::Identity() +
           lambda * ((J - a) * cross_norm_H.m[0] + cross_norm_J.transpose() * cross_norm_J);
  return H;
}

template class ElasticNeoHookeanSimpleF3x2<float>;
template class ElasticNeoHookeanSimpleF3x2<double>;

template <typename Real>
LM_DEVICE_FUNC bool ElasticNeoHookeanTetrahedron<Real>::ValidInput(const InputType &V) const {
  FEMTetrahedronDeformationGradient<Real> deformation_gradient{Dm};
  ElasticNeoHookean<Real> neo_hookean{mu, lambda};
  return deformation_gradient.ValidInput(V) && neo_hookean.ValidInput(deformation_gradient(V));
}

template <typename Real>
LM_DEVICE_FUNC typename ElasticNeoHookeanTetrahedron<Real>::OutputType ElasticNeoHookeanTetrahedron<Real>::operator()(
    const InputType &V) const {
  FEMTetrahedronDeformationGradient<Real> deformation_gradient{Dm};
  ElasticNeoHookean<Real> neo_hookean{mu, lambda};
  return neo_hookean(deformation_gradient(V));
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 12> ElasticNeoHookeanTetrahedron<Real>::Jacobian(const InputType &V) const {
  FEMTetrahedronDeformationGradient<Real> deformation_gradient{Dm};
  ElasticNeoHookean<Real> neo_hookean{mu, lambda};
  return neo_hookean.Jacobian(deformation_gradient(V)) * deformation_gradient.Jacobian(V);
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, 1, 12> ElasticNeoHookeanTetrahedron<Real>::Hessian(const InputType &V) const {
  FEMTetrahedronDeformationGradient<Real> deformation_gradient{Dm};
  ElasticNeoHookean<Real> neo_hookean{mu, lambda};
  return neo_hookean.Hessian(deformation_gradient(V)) * deformation_gradient.Jacobian(V);
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix3<Real> ElasticNeoHookeanTetrahedron<Real>::SubHessian(const InputType &V, int dim) const {
  Eigen::Vector3<Real> J;
  switch (dim) {
    case 0:
      J = {-1.0, -1.0, -1.0};
      break;
    case 1:
      J = {1.0, 0.0, 0.0};
      break;
    case 2:
      J = {0.0, 1.0, 0.0};
      break;
    case 3:
      J = {0.0, 0.0, 1.0};
      break;
    default:
      J = {0.0, 0.0, 0.0};
      break;
  }
  Eigen::Matrix3<Real> Dm_inv = Dm.inverse();
  Eigen::Matrix3<Real> Dm_inv_t = Dm_inv.transpose();
  J = Dm.transpose().inverse() * J;
  Eigen::Matrix3<Real> result = Eigen::Matrix3<Real>::Zero();
  FEMTetrahedronDeformationGradient<Real> deformation_gradient{Dm};
  ElasticNeoHookean<Real> neo_hookean{mu, lambda};
  Eigen::Matrix<Real, 9, 9> H = neo_hookean.Hessian(deformation_gradient(V)).m[0];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      result += H.block(i * 3, j * 3, 3, 3) * J(i) * J(j);
    }
  }
  return result;
}

template class ElasticNeoHookeanTetrahedron<float>;
template class ElasticNeoHookeanTetrahedron<double>;

template <typename Real>
LM_DEVICE_FUNC bool ElasticNeoHookeanSimpleTetrahedron<Real>::ValidInput(const InputType &V) const {
  FEMTetrahedronDeformationGradient<Real> deformation_gradient{Dm};
  ElasticNeoHookeanSimple<Real> neo_hookean{mu, lambda};
  return deformation_gradient.ValidInput(V) && neo_hookean.ValidInput(deformation_gradient(V));
}

template <typename Real>
LM_DEVICE_FUNC typename ElasticNeoHookeanSimpleTetrahedron<Real>::OutputType
ElasticNeoHookeanSimpleTetrahedron<Real>::operator()(const InputType &V) const {
  FEMTetrahedronDeformationGradient<Real> deformation_gradient{Dm};
  ElasticNeoHookeanSimple<Real> neo_hookean{mu, lambda};
  return neo_hookean(deformation_gradient(V));
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 12> ElasticNeoHookeanSimpleTetrahedron<Real>::Jacobian(const InputType &V) const {
  FEMTetrahedronDeformationGradient<Real> deformation_gradient{Dm};
  ElasticNeoHookeanSimple<Real> neo_hookean{mu, lambda};
  return neo_hookean.Jacobian(deformation_gradient(V)) * deformation_gradient.Jacobian(V);
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, 1, 12> ElasticNeoHookeanSimpleTetrahedron<Real>::Hessian(const InputType &V) const {
  FEMTetrahedronDeformationGradient<Real> deformation_gradient{Dm};
  ElasticNeoHookeanSimple<Real> neo_hookean{mu, lambda};
  return neo_hookean.Hessian(deformation_gradient(V)) * deformation_gradient.Jacobian(V);
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix3<Real> ElasticNeoHookeanSimpleTetrahedron<Real>::SubHessian(const InputType &V,
                                                                                         int dim) const {
  Eigen::Vector3<Real> J;
  switch (dim) {
    case 0:
      J = {-1.0, -1.0, -1.0};
      break;
    case 1:
      J = {1.0, 0.0, 0.0};
      break;
    case 2:
      J = {0.0, 1.0, 0.0};
      break;
    case 3:
      J = {0.0, 0.0, 1.0};
      break;
    default:
      J = {0.0, 0.0, 0.0};
      break;
  }
  Eigen::Matrix3<Real> Dm_inv = Dm.inverse();
  Eigen::Matrix3<Real> Dm_inv_t = Dm_inv.transpose();
  J = Dm.transpose().inverse() * J;
  Eigen::Matrix3<Real> result = Eigen::Matrix3<Real>::Zero();
  FEMTetrahedronDeformationGradient<Real> deformation_gradient{Dm};
  ElasticNeoHookeanSimple<Real> neo_hookean{mu, lambda};
  Eigen::Matrix<Real, 9, 9> H = neo_hookean.Hessian(deformation_gradient(V)).m[0];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      result += H.block(i * 3, j * 3, 3, 3) * J(i) * J(j);
    }
  }
  return result;
}

template class ElasticNeoHookeanSimpleTetrahedron<float>;
template class ElasticNeoHookeanSimpleTetrahedron<double>;

template <typename Real>
LM_DEVICE_FUNC bool ElasticNeoHookeanTriangle<Real>::ValidInput(const InputType &V) const {
  FEMTriangleDeformationGradient3x2<Real> deformation_gradient3x2{Dm};
  ElasticNeoHookeanF3x2<Real> neo_hookean_f3x2{mu, lambda};
  return deformation_gradient3x2.ValidInput(V) && neo_hookean_f3x2.ValidInput(deformation_gradient3x2(V));
}

template <typename Real>
LM_DEVICE_FUNC typename ElasticNeoHookeanTriangle<Real>::OutputType ElasticNeoHookeanTriangle<Real>::operator()(
    const InputType &V) const {
  FEMTriangleDeformationGradient3x2<Real> deformation_gradient3x2{Dm};
  ElasticNeoHookeanF3x2<Real> neo_hookean_f3x2{mu, lambda};
  return neo_hookean_f3x2(deformation_gradient3x2(V));
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 9> ElasticNeoHookeanTriangle<Real>::Jacobian(const InputType &V) const {
  FEMTriangleDeformationGradient3x2<Real> deformation_gradient3x2{Dm};
  ElasticNeoHookeanF3x2<Real> neo_hookean_f3x2{mu, lambda};
  return neo_hookean_f3x2.Jacobian(deformation_gradient3x2(V)) * deformation_gradient3x2.Jacobian(V);
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, 1, 9> ElasticNeoHookeanTriangle<Real>::Hessian(const InputType &V) const {
  FEMTriangleDeformationGradient3x2<Real> deformation_gradient3x2{Dm};
  ElasticNeoHookeanF3x2<Real> neo_hookean_f3x2{mu, lambda};
  return neo_hookean_f3x2.Hessian(deformation_gradient3x2(V)) * deformation_gradient3x2.Jacobian(V);
}

template class ElasticNeoHookeanTriangle<float>;
template class ElasticNeoHookeanTriangle<double>;

template <typename Real>
LM_DEVICE_FUNC bool ElasticNeoHookeanSimpleTriangle<Real>::ValidInput(const InputType &V) const {
  FEMTriangleDeformationGradient3x2<Real> deformation_gradient3x2{Dm};
  ElasticNeoHookeanSimpleF3x2<Real> neo_hookean_f3x2{mu, lambda};
  return deformation_gradient3x2.ValidInput(V) && neo_hookean_f3x2.ValidInput(deformation_gradient3x2(V));
}

template <typename Real>
LM_DEVICE_FUNC typename ElasticNeoHookeanSimpleTriangle<Real>::OutputType
ElasticNeoHookeanSimpleTriangle<Real>::operator()(const InputType &V) const {
  FEMTriangleDeformationGradient3x2<Real> deformation_gradient3x2{Dm};
  ElasticNeoHookeanSimpleF3x2<Real> neo_hookean_f3x2{mu, lambda};
  return neo_hookean_f3x2(deformation_gradient3x2(V));
}

template <typename Real>
LM_DEVICE_FUNC Eigen::Matrix<Real, 1, 9> ElasticNeoHookeanSimpleTriangle<Real>::Jacobian(const InputType &V) const {
  FEMTriangleDeformationGradient3x2<Real> deformation_gradient3x2{Dm};
  ElasticNeoHookeanSimpleF3x2<Real> neo_hookean_f3x2{mu, lambda};
  return neo_hookean_f3x2.Jacobian(deformation_gradient3x2(V)) * deformation_gradient3x2.Jacobian(V);
}

template <typename Real>
LM_DEVICE_FUNC HessianTensor<Real, 1, 9> ElasticNeoHookeanSimpleTriangle<Real>::Hessian(const InputType &V) const {
  FEMTriangleDeformationGradient3x2<Real> deformation_gradient3x2{Dm};
  ElasticNeoHookeanSimpleF3x2<Real> neo_hookean_f3x2{mu, lambda};
  return neo_hookean_f3x2.Hessian(deformation_gradient3x2(V)) * deformation_gradient3x2.Jacobian(V);
}

template class ElasticNeoHookeanSimpleTriangle<float>;
template class ElasticNeoHookeanSimpleTriangle<double>;
}  // namespace grassland
