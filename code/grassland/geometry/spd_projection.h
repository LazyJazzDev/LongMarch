#pragma once

#include "grassland/geometry/common.h"

namespace grassland::geometry {

template <typename Scalar, int dim>
Matrix<Scalar, dim, dim> SPDProjection(const Matrix<Scalar, dim, dim> &A) {
  Eigen::SelfAdjointEigenSolver<geometry::Matrix<Scalar, dim, dim>> eig_solver(
      A);
  const geometry::Vector<Scalar, dim> &la = eig_solver.eigenvalues();
  const geometry::Matrix<Scalar, dim, dim> &V = eig_solver.eigenvectors();
  return V * la.cwiseMax(Eigen::Vector<Scalar, dim>::Zero()).asDiagonal() *
         V.transpose();
}

}  // namespace grassland::geometry
