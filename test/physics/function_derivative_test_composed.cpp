#include "function_derivative_test.h"

TEST(Physics, FunctionDerivativeFEMTriangleDeformationGradient3x3RightMulMatrix) {
  Eigen::Matrix2<double> Dm;
  Eigen::Matrix<double, 3, 3> Fp;
  do {
    Dm = Eigen::Matrix2<double>::Random();
  } while (Dm.determinant() < 0);
  Fp = Eigen::Matrix<double, 3, 3>::Random();
  TestFunctionSet<RightMultiplyMatrix<FEMTriangleDeformationGradient3x3<double>, Eigen::Matrix<double, 3, 3>>>(
      {{Dm}, Fp});
}

TEST(Physics, FunctionDerivativeFEMTriangleNeoHookeanElement) {
  Eigen::Matrix2<double> Dm;
  Eigen::Matrix<double, 3, 3> Fp;
  do {
    Dm = Eigen::Matrix2<double>::Random();
  } while (Dm.determinant() < 0.5);
  Fp = Eigen::Matrix<double, 3, 3>::Random();
  TestFunctionSet<Compose<RightMultiplyMatrix<FEMTriangleDeformationGradient3x3<double>, Eigen::Matrix<double, 3, 3>>,
                          ElasticNeoHookean<double>>>({{{Dm}, Fp}, {1.0, 1.0}});
}
