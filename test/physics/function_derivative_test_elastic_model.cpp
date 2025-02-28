#include "function_derivative_test.h"

TEST(Physics, FunctionDerivativeElasticNeoHookean) {
  TestFunctionSet<ElasticNeoHookean<double>>();
}

TEST(Physics, FunctionDerivativeElasticNeoHookeanSimple) {
  TestFunctionSet<ElasticNeoHookeanSimple<double>>();
}

TEST(Physics, FunctionDerivativeElasticNeoHookeanF3x2) {
  TestFunctionSet<ElasticNeoHookeanF3x2<double>>();
}

TEST(Physics, FunctionDerivativeElasticNeoHookeanSimpleF3x2) {
  TestFunctionSet<ElasticNeoHookeanSimpleF3x2<double>>();
}

TEST(Physics, FunctionDerivativeElasticNeoHookeanTetrahedron) {
  Eigen::Matrix3<double> Dm;
  do {
    Dm = Eigen::Matrix3<double>::Random();
  } while (Dm.determinant() < 0);
  TestFunctionSet<ElasticNeoHookeanTetrahedron<double>>({1.0, 1.0, Dm});
}

TEST(Physics, FunctionDerivativeElasticNeoHookeanSimpleTetrahedron) {
  Eigen::Matrix3<double> Dm;
  do {
    Dm = Eigen::Matrix3<double>::Random();
  } while (Dm.determinant() < 0);
  TestFunctionSet<ElasticNeoHookeanSimpleTetrahedron<double>>({1.0, 1.0, Dm});
}

TEST(Physics, FunctionDerivativeElasticNeoHookeanTriangle) {
  Eigen::Matrix2<double> Dm;
  do {
    Dm = Eigen::Matrix2<double>::Random();
  } while (Dm.determinant() < 0);
  TestFunctionSet<ElasticNeoHookeanTriangle<double>>({1.0, 1.0, Dm});
}

TEST(Physics, FunctionDerivativeElasticNeoHookeanSimpleTriangle) {
  Eigen::Matrix2<double> Dm;
  do {
    Dm = Eigen::Matrix2<double>::Random();
  } while (Dm.determinant() < 0);
  TestFunctionSet<ElasticNeoHookeanSimpleTriangle<double>>({1.0, 1.0, Dm});
}
