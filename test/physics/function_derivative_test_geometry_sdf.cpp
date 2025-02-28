#include "function_derivative_test.h"

TEST(Physics, FunctionDerivativePointSDF) {
  PointSDF<double> f;
  f.position = Eigen::Vector3d::Random();
  TestFunctionSet<PointSDF<double>>(f);
}

TEST(Physics, FunctionDerivativeSphereSDF) {
  SphereSDF<double> f;
  f.center = Eigen::Vector3d::Random();
  f.radius = 3.0;
  TestFunctionSet<SphereSDF<double>>(f);
}

TEST(Physics, FunctionDerivativeSegmentSDF) {
  for (int i = 0; i < 100; i++) {
    SegmentSDF<double> f;
    f.A = Eigen::Vector3d::Random();
    f.B = Eigen::Vector3d::Random();
    TestFunctionSet<SegmentSDF<double>>(f, 1);
  }
}

TEST(Physics, FunctionDerivativeCapsuleSDF) {
  for (int i = 0; i < 100; i++) {
    CapsuleSDF<double> f;
    f.A = Eigen::Vector3d::Random();
    f.B = Eigen::Vector3d::Random();
    f.radius = 3.0;
    TestFunctionSet<CapsuleSDF<double>>(f, 1);
  }
}

TEST(Physics, FunctionDerivativeCubeSDF) {
  for (int i = 0; i < 100; i++) {
    CubeSDF<double> f;
    f.center = {0.0, 0.0, 0.0};
    f.size = 0.1;
    TestFunctionSet<CubeSDF<double>>(f, 1);
  }
}

TEST(Physics, FunctionDerivativePlaneSDF) {
  for (int i = 0; i < 100; i++) {
    PlaneSDF<double> f;
    f.normal = Eigen::Vector3<double>::Random().normalized();
    f.d = Eigen::Matrix<double, 1, 1>::Random().value();
    TestFunctionSet<PlaneSDF<double>>(f, 1);
  }
}

TEST(Physics, FunctionDerivativeLineSDF) {
  for (int i = 0; i < 100; i++) {
    LineSDF<double> f;
    f.origin = Eigen::Vector3<double>::Random();
    f.direction = Eigen::Vector3<double>::Random().normalized();
    TestFunctionSet<LineSDF<double>>(f, 1);
  }
}
