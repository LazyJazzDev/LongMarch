#pragma once
#include "cao_di/math/math_util.h"

namespace CD {

template <typename Scalar>
Scalar EvaluateLinearPolynomial(Scalar a, Scalar b, Scalar x) {
  return a * x + b;
}

template <typename Scalar>
Scalar EvaluateQuadraticPolynomial(Scalar a, Scalar b, Scalar c, Scalar x) {
  return a * x * x + b * x + c;
}

template <typename Scalar>
Scalar EvaluateCubicPolynomial(Scalar a, Scalar b, Scalar c, Scalar d, Scalar x) {
  return a * x * x * x + b * x * x + c * x + d;
}

template <typename Scalar>  // ax + b = 0
void SolveLinearPolynomialLimitedRange(Scalar a,
                                       Scalar b,
                                       Scalar *roots,
                                       int *num_roots,
                                       Scalar low_root = 0.0,
                                       Scalar high_root = 1.0) {
  if (a == 0) {
    *num_roots = 0;
    return;
  }
  Scalar root = -b / a;
  if (root >= low_root && root <= high_root) {
    roots[0] = root;
    *num_roots = 1;
  } else {
    *num_roots = 0;
  }
}

template <typename Scalar>
bool BinarySearchRoot(Scalar a, Scalar b, Scalar c, Scalar low_root, Scalar high_root, Scalar *root) {
  Scalar low_value = EvaluateQuadraticPolynomial(a, b, c, low_root);
  Scalar high_value = EvaluateQuadraticPolynomial(a, b, c, high_root);
  if (low_value * high_value > 0) {
    return false;
  }
  while (high_root - low_root > Eps<Scalar>() * 1e-2) {
    Scalar mid_root = (low_root + high_root) / 2;
    Scalar mid_value = EvaluateQuadraticPolynomial(a, b, c, mid_root);
    if (mid_value * low_value > 0) {
      low_root = mid_root;
      low_value = mid_value;
    } else {
      high_root = mid_root;
      high_value = mid_value;
    }
  }
  *root = (low_root + high_root) / 2;
  return true;
}

template <typename Scalar>
void CubicPolynomialNewtonIteration(Scalar a, Scalar b, Scalar c, Scalar d, Scalar *root) {
  Scalar x = *root;
  Scalar f = EvaluateCubicPolynomial(a, b, c, d, x);
  Scalar df = 3 * a * x * x + 2 * b * x + c;
  *root = x - f / df;
}

template <typename Scalar>
bool BinarySearchRoot(Scalar a, Scalar b, Scalar c, Scalar d, Scalar low_root, Scalar high_root, Scalar *root) {
  Scalar low_value = EvaluateCubicPolynomial(a, b, c, d, low_root);
  Scalar high_value = EvaluateCubicPolynomial(a, b, c, d, high_root);
  if (low_value * high_value > 0) {
    return false;
  }
  while (high_root - low_root > Eps<Scalar>() * 1e-2) {
    Scalar mid_root = (low_root + high_root) / 2;
    Scalar mid_value = EvaluateCubicPolynomial(a, b, c, d, mid_root);
    if (mid_value * low_value > 0) {
      low_root = mid_root;
      low_value = mid_value;
    } else {
      high_root = mid_root;
      high_value = mid_value;
    }
  }
  *root = (low_root + high_root) / 2;
  CubicPolynomialNewtonIteration(a, b, c, d, root);
  return true;
}

template <typename Scalar>  // ax^2 + bx + c
void SolveQuadraticPolynomialLimitedRange(Scalar a,
                                          Scalar b,
                                          Scalar c,
                                          Scalar *roots,
                                          int *num_roots,
                                          Scalar low_root = 0.0,
                                          Scalar high_root = 1.0) {
  // use linear equation solver to binary search
  if (a != 0) {
    Scalar deri_a = 2 * a;
    Scalar deri_b = b;
    Scalar endpoints[3] = {low_root};
    int num_endpoints = 0;
    SolveLinearPolynomialLimitedRange(deri_a, deri_b, endpoints + 1, &num_endpoints, low_root, high_root);
    num_endpoints++;
    endpoints[num_endpoints++] = high_root;
    *num_roots = 0;
    for (int i = 0; i < num_endpoints - 1; i++) {
      Scalar root;
      if (BinarySearchRoot(a, b, c, endpoints[i], endpoints[i + 1], &root)) {
        roots[*num_roots] = root;
        (*num_roots)++;
      }
    }
  } else {
    SolveLinearPolynomialLimitedRange(b, c, roots, num_roots, low_root, high_root);
  }
}

template <typename Scalar>  // ax^3 + bx^2 + cx + d
void SolveCubicPolynomialLimitedRange(Scalar a,
                                      Scalar b,
                                      Scalar c,
                                      Scalar d,
                                      Scalar *roots,
                                      int *num_roots,
                                      Scalar low_root = 0.0,
                                      Scalar high_root = 1.0) {
  if (a == 0) {
    SolveQuadraticPolynomialLimitedRange(b, c, d, roots, num_roots, low_root, high_root);
    return;
  }
  Scalar derivative_a = 3 * a;
  Scalar derivative_b = 2 * b;
  Scalar derivative_c = c;
  Scalar endpoints[4] = {low_root};
  int num_endpoints = 0;
  SolveQuadraticPolynomialLimitedRange(derivative_a, derivative_b, derivative_c, endpoints + 1, &num_endpoints,
                                       low_root, high_root);
  num_endpoints++;
  endpoints[num_endpoints++] = high_root;
  *num_roots = 0;
  for (int i = 0; i < num_endpoints - 1; i++) {
    Scalar root;
    if (BinarySearchRoot(a, b, c, d, endpoints[i], endpoints[i + 1], &root)) {
      roots[*num_roots] = root;
      (*num_roots)++;
    }
  }
}
}  // namespace CD
