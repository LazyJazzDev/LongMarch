#pragma once
#include "gtest/gtest.h"
#include "long_march.h"
#include "random"

using namespace long_march;

template <class Scalar>
void TestDiagonalizeOperator() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Scalar> dis(-1, 1);

  geometry::Matrix2<Scalar> A, Q;
  Scalar a, b, c;
  a = dis(gen);
  b = dis(gen);
  c = dis(gen);
  A << a, b, b, c;
  Q = geometry::DiagonalizeOperator(a, b, c);
  geometry::Matrix2<Scalar> D = Q * A * Q.transpose();
  EXPECT_NEAR(D(1, 0), 0.0, geometry::Eps<Scalar>());
  EXPECT_NEAR(D(0, 1), 0.0, geometry::Eps<Scalar>());
  if (fabs(D(0, 1)) > geometry::Eps<Scalar>()) {
    std::cout << "A = " << A << std::endl;
    std::cout << "Q = " << Q << std::endl;
    std::cout << "D = " << D << std::endl;
    std::cout << "Q^T A Q = " << Q.transpose() * A * Q << std::endl;
  }
}

template <class Scalar>
void BatchedTest() {
  for (int i = 0; i < 100000; i++) {
    TestDiagonalizeOperator<Scalar>();
  }
}

TEST(Geometry, SPDProjection) {
  using real = float;
  constexpr int dim = 3;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<real> dist(-1, 1);
  geometry::Matrix<real, dim, dim> A;
  for (int i = 0; i < dim; i++) {
    for (int j = i; j < dim; j++) {
      real v = dist(gen);
      A(j, i) = A(i, j) = v;
    }
  }
  geometry::Matrix<real, dim, dim> eigen_vectors;
  geometry::Vector<real, dim> eigen_values;
  geometry::SymmetricMatrixDecomposition(A, eigen_vectors, eigen_values);

  BatchedTest<float>();
  BatchedTest<double>();
}
