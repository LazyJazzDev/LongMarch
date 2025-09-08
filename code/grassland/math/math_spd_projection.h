#pragma once

#include "grassland/math/math_util.h"

namespace CD {

template <typename Scalar, int dim>
Matrix<Scalar, dim, dim> SPDProjection(const Matrix<Scalar, dim, dim> &A) {
  Eigen::SelfAdjointEigenSolver<Matrix<Scalar, dim, dim>> eig_solver(A);
  const Vector<Scalar, dim> &la = eig_solver.eigenvalues();
  const Matrix<Scalar, dim, dim> &V = eig_solver.eigenvectors();
  return V * la.cwiseMax(Vector<Scalar, dim>::Zero()).asDiagonal() * V.transpose();
}

}  // namespace CD
