#pragma once
#include "grassland/geometry/common.h"

namespace grassland::geometry {

template <typename Scalar>
Matrix2<Scalar> DiagonalizeOperator(Scalar a00, Scalar a01, Scalar a11) {
  if (a01 < 0.0) {
    a00 = -a00;
    a11 = -a11;
    a01 = -a01;
  }
  a00 -= a11;
  a11 = sqrt(a00 * a00 + 4.0 * a01 * a01);
  a01 = fabs(2.0 * a01 / a11);
  a11 = a00 / a11;
  if (1.0 - fabs(a11) > Eps<Scalar>()) {
    a00 = sqrt((1.0 + a11) * 0.5);
    a01 = sqrt((1.0 - a11) * 0.5);
  } else {
    if (a11 > 0) {
      a00 = sqrt((1.0 + a11) * 0.5);
      a01 = a01 * 0.5;
    } else {
      a00 = a01 * 0.5;
      a01 = sqrt((1.0 - a11) * 0.5);
    }
  }
  Matrix2<Scalar> result;
  result << a00, a01, -a01, a00;
  return result;
}

template <typename Scalar, int dim>
void SymmetricMatrixDecomposition(const Matrix<Scalar, dim, dim> &A,
                                  Matrix<Scalar, dim, dim> &eigen_vectors,
                                  Vector<Scalar, dim> &eigen_values) {
  Matrix<Scalar, dim, dim> S = (A + A.transpose()) * 0.5;
  eigen_vectors = Matrix<Scalar, dim, dim>::Identity();
  Scalar temp[dim * 2];
  for (int _i = 1; _i < 2; _i++) {
    for (int _j = 0; _j < dim - _i; _j++) {
      int i = _i - 1;
      int j = dim - 1 - _j;
      Matrix2<Scalar> Q_part = DiagonalizeOperator(S(i, i), S(i, j), S(j, j));
      Eigen::Map<Matrix<Scalar, dim, 2>> cols(temp);
      Eigen::Map<Matrix<Scalar, 2, dim>> rows(temp);
      cols.col(0) = S.col(i);
      cols.col(1) = S.col(j);
      cols = cols * Q_part.transpose();
      S.col(i) = cols.col(0);
      S.col(j) = cols.col(1);

      rows.row(0) = S.row(i);
      rows.row(1) = S.row(j);
      rows = Q_part * rows;
      S.row(i) = rows.row(0);
      S.row(j) = rows.row(1);

      cols.col(0) = eigen_vectors.col(i);
      cols.col(1) = eigen_vectors.col(j);
      cols = cols * Q_part.transpose();
      eigen_vectors.col(i) = cols.col(0);
      eigen_vectors.col(j) = cols.col(1);
      //      eigen_vectors.col(index) = eigen_vectors.col(index) *
      //      Q_part.transpose();
    }
  }
  eigen_values = S.diagonal();
  std::cout << eigen_values << std::endl;
  std::cout << eigen_vectors << std::endl;
  std::cout << A << std::endl;
  std::cout << S << std::endl;
  std::cout << eigen_vectors * S * eigen_vectors.transpose();
}

}  // namespace grassland::geometry
