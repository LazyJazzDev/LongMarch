#include "function_derivative_test.h"

TEST(Physics, FunctionDerivativeDeterminant3) {
  TestFunctionSet<Determinant3<double>>();
}

TEST(Physics, FunctionDerivativeLogDeterminant3) {
  TestFunctionSet<LogDeterminant3<double>>();
}

TEST(Physics, FunctionDerivativeLogSquareDeterminant3) {
  TestFunctionSet<LogSquareDeterminant3<double>>();
}

TEST(Physics, FunctionDerivativeVecLength) {
  TestFunctionSet<VecLength<double, 3>>();
  TestFunctionSet<VecLength<double, 4>>();
  TestFunctionSet<VecLength<double, 5>>();
}

TEST(Physics, FunctionDerivativeVecNormalized) {
  TestFunctionSet<VecNormalized<double, 3>>();
  TestFunctionSet<VecNormalized<double, 4>>();
  TestFunctionSet<VecNormalized<double, 5>>();
}

TEST(Physics, FunctionDerivativeCross3) {
  TestFunctionSet<Cross3<double>>();
}

TEST(Physics, FunctionDerivativeDot) {
  TestFunctionSet<Dot<double>>();
}

TEST(Physics, FunctionDerivativeCrossNormalized) {
  TestFunctionSet<CrossNormalized<double>>();
}

TEST(Physics, FunctionDerivativeAtan2) {
  TestFunctionSet<Atan2<double>>();
}

TEST(Physics, FunctionDerivativeCrossNorm) {
  TestFunctionSet<CrossNorm<double>>();
}
