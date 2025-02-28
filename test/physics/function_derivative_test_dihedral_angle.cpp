#include "function_derivative_test.h"

TEST(Physics, FunctionDerivativeDihedralAngleAssistEdgesToNormalsAxis) {
  TestFunctionSet<DihedralAngleAssistEdgesToNormalsAxis<double>>();
}

TEST(Physics, FunctionDerivativeDihedralAngleAssistNormalsAxisToSinCosTheta) {
  TestFunctionSet<DihedralAngleAssistNormalsAxisToSinCosTheta<double>>();
}

TEST(Physics, FunctionDerivativeDihedralAngleByEdges) {
  TestFunctionSet<DihedralAngleByEdges<double>>();
}

TEST(Physics, FunctionDerivativeDihedralAngleAssistVerticesToEdges) {
  TestFunctionSet<DihedralAngleAssistVerticesToEdges<double>>();
}

TEST(Physics, FunctionDerivativeDihedralAngleByVertices) {
  TestFunctionSet<DihedralAngleByVertices<double>>();
}

TEST(Physics, FunctionDerivativeDihedralEnergy) {
  DihedralEnergy<double> f;
  std::random_device rd;
  for (int i = 0; i < 100; i++) {
    f.rest_angle = std::uniform_real_distribution<double>(-glm::pi<double>() * 0.5, glm::pi<double>() * 0.5)(rd);
    TestFunctionSet(f, 1);
  }
}

TEST(Physics, FunctionDerivativeDihedralAngle) {
  // Eigen::Vector<double, 12> x;
  // x << 0, 2, 2, 0, 0, 2, 0, 0, -2, -2, 0, -2;
  // TestFunctionSetWithInput<DihedralAngle<double>>({}, x);
  TestFunctionSet<DihedralAngle<double>>({});
}
