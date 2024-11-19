#include <glm/ext/scalar_constants.hpp>

#include "cmath"
#include "grassland/physics/physics.h"
#include "gtest/gtest.h"
#include "iostream"
#include "long_march.h"
#include "random"

using namespace long_march;

template <typename FunctionSet = Determinant3<double>>
void TestFunctionSet(FunctionSet f = FunctionSet{}, const int test_cnt = 100) {
  using Real = typename FunctionSet::Scalar;
  for (int i = 0; i < test_cnt; i++) {
    using InputVec =
        Eigen::Vector<Real, FunctionSet::InputType::SizeAtCompileTime>;
    using OutputVec =
        Eigen::Vector<Real, FunctionSet::OutputType::SizeAtCompileTime>;

    using JacobiType =
        Eigen::Matrix<Real, FunctionSet::OutputType::SizeAtCompileTime,
                      FunctionSet::InputType::SizeAtCompileTime>;

    auto InputVecToInputType = [](const InputVec &x) ->
        typename FunctionSet::InputType {
          return Eigen::Map<const typename FunctionSet::InputType>(x.data());
        };

    auto OutputTypeToOutputVec =
        [](const typename FunctionSet::OutputType &y) -> OutputVec {
      return Eigen::Map<const OutputVec>(y.data());
    };

    InputVec x = InputVec::Random();

    while (!f.ValidInput(InputVecToInputType(x))) {
      x = InputVec::Random();
    }

    Real eps = algebra::Eps<Real>();
    OutputVec y = OutputTypeToOutputVec(f(InputVecToInputType(x)));
    JacobiType J = f.Jacobian(InputVecToInputType(x));
    JacobiType J_finite_diff;
    J_finite_diff.setZero();

    for (int j = 0; j < x.size(); j++) {
      InputVec x_plus = x;
      x_plus[j] += eps;
      OutputVec y_plus = OutputTypeToOutputVec(f(InputVecToInputType(x_plus)));

      InputVec x_minus = x;
      x_minus[j] -= eps;
      OutputVec y_minus =
          OutputTypeToOutputVec(f(InputVecToInputType(x_minus)));

      OutputVec dy = (y_plus - y_minus) / (2 * eps);

      J_finite_diff.col(j) = dy;
    }

    // std::cout << std::fixed;
    // std::cout << "x:\n" << Eigen::Map<FunctionSet::InputType>(x.data()) <<
    // std::endl; std::cout << "y:\n" <<
    // Eigen::Map<FunctionSet::OutputType>(y.data()) << std::endl; std::cout <<
    // "J: \n" << J << std::endl; std::cout << "J_finite_diff: \n" <<
    // J_finite_diff << std::endl;

    // Compare J and J_finite_diff
    for (int j = 0; j < J.size(); j++) {
      EXPECT_NEAR(J(j), J_finite_diff(j),
                  fmax(fabs(sqrt(eps) * J(j)), sqrt(eps)));
    }

    using HessianType =
        HessianTensor<Real, FunctionSet::OutputType::SizeAtCompileTime,
                      FunctionSet::InputType::SizeAtCompileTime>;
    HessianType H = f.Hessian(InputVecToInputType(x));
    HessianType H_finite_diff;

    for (int j = 0; j < x.size(); j++) {
      InputVec x_plus = x;
      x_plus[j] += eps;
      JacobiType J_plus = f.Jacobian(InputVecToInputType(x_plus));

      InputVec x_minus = x;
      x_minus[j] -= eps;
      JacobiType J_minus = f.Jacobian(InputVecToInputType(x_minus));

      JacobiType dJ = (J_plus - J_minus) / (2 * eps);

      for (int k = 0; k < dJ.rows(); k++) {
        for (int l = 0; l < dJ.cols(); l++) {
          H_finite_diff.m[k](j, l) = dJ(k, l);
        }
      }
    }

    // std::cout << std::fixed;
    // std::cout << "x:\n" << x << std::endl;
    // std::cout << "H: \n" << H << std::endl;
    // std::cout << "H_finite_diff: \n" << H_finite_diff << std::endl;
    // std::cout << "Diff: \n" << H - H_finite_diff << std::endl;
    // std::cout.flush();

    for (int j = 0; j < OutputVec::SizeAtCompileTime; j++) {
      for (int k = 0; k < InputVec::SizeAtCompileTime; k++) {
        for (int l = 0; l < InputVec::SizeAtCompileTime; l++) {
          EXPECT_NEAR(H.m[j](k, l), H_finite_diff.m[j](k, l),
                      fmax(fabs(sqrt(eps) * H.m[j](k, l)), sqrt(eps)));
        }
      }
    }
  }
}
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

TEST(Physics, FunctionDerivativeElasticNeoHookean) {
  TestFunctionSet<ElasticNeoHookean<double>>();
}

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
    f.rest_angle = std::uniform_real_distribution<double>(
        -glm::pi<double>() * 0.5, glm::pi<double>() * 0.5)(rd);
    TestFunctionSet(f, 1);
  }
}

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
