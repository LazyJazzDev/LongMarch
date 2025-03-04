#include "gtest/gtest.h"
#include "long_march.h"
#include "random"
#include "vector"

using namespace long_march;

template <typename Scalar>
std::vector<Scalar> PolynomialMul(const std::vector<Scalar> &poly1, const std::vector<Scalar> &poly2) {
  std::vector<Scalar> result(poly1.size() + poly2.size() - 1, 0);
  for (size_t i = 0; i < poly1.size(); ++i) {
    for (size_t j = 0; j < poly2.size(); ++j) {
      result[i + j] += poly1[i] * poly2[j];
    }
  }
  return result;
}

template <typename Scalar>
Scalar PolynomialEval(const std::vector<Scalar> &poly, Scalar x) {
  Scalar result = 0;
  Scalar x_i = 1;
  for (size_t i = 0; i < poly.size(); ++i) {
    result += poly[i] * x_i;
    x_i *= x;
  }
  Scalar deri = 0;
  x_i = 1;
  for (size_t i = 1; i < poly.size(); ++i) {
    deri += i * poly[i] * x_i;
    x_i *= x;
  }
  if (deri == 0) {
    return 0.0;
  }
  return fabs(result / deri) / fmax(fabs(x), Scalar(1));
}

template <typename Scalar>
std::vector<Scalar> RandomUnsolvableQuadraticPolynomial() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Scalar> dis(-1, 1);
  std::vector<Scalar> poly(3);
  do {
    for (size_t i = 0; i < 3; ++i) {
      poly[i] = dis(gen);
    }
  } while (poly[1] * poly[1] - 4 * poly[0] * poly[2] >= -0.01 || fabs(poly[2]) < 0.5);
  return poly;
}

template <typename Scalar>
std::vector<Scalar> RandomLinearPolynomial() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<Scalar> poly(2);
  poly[0] = -std::uniform_real_distribution<Scalar>(Eps<Scalar>(), 1.0 - Eps<Scalar>())(gen);
  poly[1] = 1.0;
  Scalar scale = 0.0;
  do {
    scale = std::uniform_real_distribution<Scalar>(-1.0, 1.0)(gen);
  } while (fabs(scale) < 0.01);
  for (int i = 0; i < poly.size(); i++) {
    poly[i] *= scale;
  }
  return poly;
}

template <typename Scalar>
std::vector<Scalar> RandomSolvableQuadraticPolynomial() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Scalar> dis(Eps<Scalar>(), 1.0 - Eps<Scalar>());
  Scalar x0, x1;
  do {
    x0 = dis(gen);
    x1 = dis(gen);
  } while (fabs(x0 - x1) < 0.01);
  std::vector<Scalar> poly = PolynomialMul<Scalar>({-x0, 1.0}, {-x1, 1.0});
  Scalar scale = 0.0;
  do {
    scale = std::uniform_real_distribution<Scalar>(-1.0, 1.0)(gen);
  } while (fabs(scale) < 0.01);
  for (int i = 0; i < poly.size(); i++) {
    poly[i] *= scale;
  }
  return poly;
}

template <typename Scalar>
Scalar MaxCoeff(const std::vector<Scalar> &poly, Scalar x = 1) {
  Scalar max_coeff = 0;
  Scalar x_i = 1;
  for (int i = 0; i < poly.size(); i++) {
    auto coeff = poly[i];
    max_coeff = std::max(max_coeff, fabs(coeff * x * i));
  }
  return max_coeff;
}

template <typename Scalar>
void TestUnsolvableCubicPolynomialQuadratic() {
  auto poly = RandomUnsolvableQuadraticPolynomial<Scalar>();
  poly.emplace_back(0);
  std::vector<Scalar> roots(3);
  int num_roots = 0;
  SolveCubicPolynomialLimitedRange(poly[3], poly[2], poly[1], poly[0], roots.data(), &num_roots);
  EXPECT_EQ(num_roots, 0);
}

template <typename Scalar>
void TestUnsolvableCubicPolynomialConstant() {
  auto poly = std::vector<Scalar>{1, 0, 0, 0};
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Scalar> dis(-1, 1);
  do {
    poly[0] = dis(gen);
  } while (poly[0] == 0);
  std::vector<Scalar> roots(3);
  int num_roots = 0;
  SolveCubicPolynomialLimitedRange(poly[3], poly[2], poly[1], poly[0], roots.data(), &num_roots);
  EXPECT_EQ(num_roots, 0);
}

template <typename Scalar>
void TestSingleRootCubicPolynomialCubic() {
  auto poly0 = RandomLinearPolynomial<Scalar>();
  auto poly1 = RandomUnsolvableQuadraticPolynomial<Scalar>();
  auto poly = PolynomialMul(poly0, poly1);
  std::vector<Scalar> roots(3);
  int num_roots = 0;
  SolveCubicPolynomialLimitedRange(poly[3], poly[2], poly[1], poly[0], roots.data(), &num_roots);
  EXPECT_EQ(num_roots, 1);
  EXPECT_NEAR(PolynomialEval(poly, roots[0]), 0, Eps<Scalar>());
  if (num_roots != 1 || fabs(PolynomialEval(poly, roots[0])) > Eps<Scalar>()) {
    std::cout << "Poly: " << poly[3] << "x^3 + " << poly[2] << "x^2 + " << poly[1] << "x + " << poly[0] << std::endl;
    std::cout << "Root: " << roots[0] << std::endl;
    std::cout << "Eval: " << PolynomialEval(poly, roots[0]) << std::endl;
  }
}

template <typename Scalar>
void TestSingleRootCubicPolynomialLinear() {
  auto poly = RandomLinearPolynomial<Scalar>();
  poly.emplace_back(0);
  poly.emplace_back(0);
  std::vector<Scalar> roots(3);
  int num_roots = 0;
  SolveCubicPolynomialLimitedRange(poly[3], poly[2], poly[1], poly[0], roots.data(), &num_roots);
  EXPECT_EQ(num_roots, 1);
  EXPECT_NEAR(PolynomialEval(poly, roots[0]), 0, Eps<Scalar>());
}

template <typename Scalar>
void TestDoubleRootCubicPolynomial() {
  auto poly = RandomSolvableQuadraticPolynomial<Scalar>();
  poly.emplace_back(0);
  std::vector<Scalar> roots(3);
  int num_roots = 0;
  SolveCubicPolynomialLimitedRange(poly[3], poly[2], poly[1], poly[0], roots.data(), &num_roots);
  EXPECT_EQ(num_roots, 2);
  EXPECT_NEAR(PolynomialEval(poly, roots[0]), 0, Eps<Scalar>());
  EXPECT_NEAR(PolynomialEval(poly, roots[1]), 0, Eps<Scalar>());

  if (num_roots != 2 || fabs(PolynomialEval(poly, roots[0])) > Eps<Scalar>() ||
      fabs(PolynomialEval(poly, roots[1])) > Eps<Scalar>()) {
    std::cout << "Poly: " << poly[3] << "x^3 + " << poly[2] << "x^2 + " << poly[1] << "x + " << poly[0] << std::endl;
    std::cout << "Root: ";
    for (int i = 0; i < num_roots; ++i) {
      std::cout << roots[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Eval: ";
    for (int i = 0; i < num_roots; ++i) {
      std::cout << PolynomialEval(poly, roots[i]) << " ";
    }
    std::cout << std::endl;
  }
}

template <typename Scalar>
void TestRandomCubicPolynomial() {
  std::vector<Scalar> poly(4);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Scalar> dis(-1, 1);
  do {
    for (size_t i = 0; i < 4; ++i) {
      poly[i] = dis(gen);
    }
  } while (fabs(poly[3]) < 0.5);
  std::vector<Scalar> roots(3);
  int num_roots = 0;
  SolveCubicPolynomialLimitedRange(poly[3], poly[2], poly[1], poly[0], roots.data(), &num_roots);
  for (int i = 0; i < num_roots; ++i) {
    EXPECT_NEAR(PolynomialEval(poly, roots[i]), 0, Eps<Scalar>());
    if (isnan(PolynomialEval(poly, roots[i])) || fabs(PolynomialEval(poly, roots[i])) > Eps<Scalar>()) {
      std::cout << "Poly: " << poly[3] << "x^3 + " << poly[2] << "x^2 + " << poly[1] << "x + " << poly[0] << std::endl;
      std::cout << "Root: ";
      for (int j = 0; j < num_roots; ++j) {
        std::cout << roots[j] << " ";
      }
      std::cout << std::endl;
      std::cout << "Eval: ";
      for (int j = 0; j < num_roots; ++j) {
        std::cout << PolynomialEval(poly, roots[j]) << " ";
      }
      std::cout << std::endl;
    }
  }
}

template <typename Scalar>
void BatchedTest() {
  for (int i = 0; i < 100000; i++) {
    TestUnsolvableCubicPolynomialQuadratic<Scalar>();
    TestUnsolvableCubicPolynomialConstant<Scalar>();
    TestSingleRootCubicPolynomialCubic<Scalar>();
    TestSingleRootCubicPolynomialLinear<Scalar>();
    TestDoubleRootCubicPolynomial<Scalar>();
    //        TestRandomCubicPolynomial<Scalar>();
  }
}

TEST(Math, CubicPolynomialSolve) {
  BatchedTest<float>();
  BatchedTest<double>();
}
