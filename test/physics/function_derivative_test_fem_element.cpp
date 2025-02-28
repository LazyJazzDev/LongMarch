#include "function_derivative_test.h"

TEST(Physics, FunctionDerivativeFEMTetrahedronDeformationGradient) {
  Eigen::Matrix3<double> Dm;
  do {
    Dm = Eigen::Matrix3<double>::Random();
  } while (Dm.determinant() < 0);
  TestFunctionSet<FEMTetrahedronDeformationGradient<double>>({Dm});
}

TEST(Physics, FunctionDerivativeFEMDeformationGradient3x2To3x3) {
  TestFunctionSet<FEMDeformationGradient3x2To3x3<double>>();
}

TEST(Physics, FunctionDerivativeFEMTriangleDeformationGradient3x2) {
  Eigen::Matrix2<double> Dm;
  do {
    Dm = Eigen::Matrix2<double>::Random();
  } while (Dm.determinant() < 0);
  TestFunctionSet<FEMTriangleDeformationGradient3x2<double>>({Dm});
}

TEST(Physics, FunctionDerivativeFEMTriangleDeformationGradient3x3) {
  Eigen::Matrix2<double> Dm;
  do {
    Dm = Eigen::Matrix2<double>::Random();
  } while (Dm.determinant() < 0);
  TestFunctionSet<FEMTriangleDeformationGradient3x3<double>>({Dm});
}
