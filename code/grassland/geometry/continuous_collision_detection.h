#pragma once
#include "grassland/geometry/common.h"

namespace grassland::geometry {

namespace {
template <typename Scalar>
LM_DEVICE_FUNC void PrivateSort(Scalar *a, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (a[i] > a[j]) {
        Scalar temp = a[i];
        a[i] = a[j];
        a[j] = temp;
      }
    }
  }
}
}  // namespace

template <typename Scalar>
LM_DEVICE_FUNC void ThirdOrderVolumetricPolynomial(const Vector3<Scalar> &p0,
                                                   const Vector3<Scalar> &p1,
                                                   const Vector3<Scalar> &p2,
                                                   const Vector3<Scalar> &v0,
                                                   const Vector3<Scalar> &v1,
                                                   const Vector3<Scalar> &v2,
                                                   Scalar *polynomial_terms) {
  // V(t) = (p0 + v0 * t) dot ((p1 + v1 * t) cross (p2 + v2 * t))
  Vector3<Scalar> cross_constant = p1.cross(p2);
  Vector3<Scalar> cross_linear = v1.cross(p2) + p1.cross(v2);
  Vector3<Scalar> cross_quadratic = v1.cross(v2);
  Scalar constant = p0.dot(cross_constant);
  Scalar linear = p0.dot(cross_linear) + v0.dot(cross_constant);
  Scalar quadratic = v0.dot(cross_linear) + p0.dot(cross_quadratic);
  Scalar cubic = v0.dot(cross_quadratic);
  polynomial_terms[0] = constant;
  polynomial_terms[1] = linear;
  polynomial_terms[2] = quadratic;
  polynomial_terms[3] = cubic;
}

template <typename Scalar>
LM_DEVICE_FUNC void SolveCubicPolynomial(Scalar a,
                                         Scalar b,
                                         Scalar c,
                                         Scalar d,
                                         Scalar *roots,
                                         int *num_roots) {
  Scalar x;
  Scalar y;
  Scalar deri;
  if (fabs(a) > 0) {
    Scalar p = b / a;
    Scalar q = c / a;
    Scalar r = d / a;
    Scalar a0 = q - p * p / 3;
    Scalar b0 = 2 * p * p * p / 27 - p * q / 3 + r;
    Scalar c0 = b0 * b0 / 4 + a0 * a0 * a0 / 27;
    const Scalar d0 = 3.14159265358979323846;
    if (c0 > 0) {
      *num_roots = 1;
      x = -p / 3 + cbrt(-b0 / 2 + sqrt(c0)) + cbrt(-b0 / 2 - sqrt(c0));
      y = a * x * x * x + b * x * x + c * x + d;
      deri = 3 * a * x * x + 2 * b * x + c;
      if (deri != 0) {
        x = x - y / deri;
      }
      roots[0] = x;
    } else {
      Scalar theta = acos(-b0 / 2 / sqrt(-a0 * a0 * a0 / 27));
      *num_roots = 3;
      x = 2 * sqrt(-a0 / 3) * cos(theta / 3) - p / 3;
      roots[0] = x;
      x = 2 * sqrt(-a0 / 3) * cos((theta + 2 * d0) / 3) - p / 3;
      roots[1] = x;
      x = 2 * sqrt(-a0 / 3) * cos((theta + 4 * d0) / 3) - p / 3;
      roots[2] = x;
    }
  } else {
    // degenerate to quadratic
    if (fabs(b) > 0) {
      a = c * c - 4 * b * d;
      if (a < 0) {
        *num_roots = 0;
      } else {
        *num_roots = 2;
        roots[0] = (-c + sqrt(a)) / 2 / b;
        roots[1] = (-c - sqrt(a)) / 2 / b;
      }
    } else {
      if (fabs(c) > 0) {
        *num_roots = 1;
        roots[0] = -d / c;
      } else {
        *num_roots = 0;
      }
    }
  }
}

template <typename Scalar>
LM_DEVICE_FUNC bool EdgeEdgeIntersection(const Vector3<Scalar> &p0,
                                         const Vector3<Scalar> &p1,
                                         const Vector3<Scalar> &p2,
                                         const Vector3<Scalar> &p3,
                                         Scalar *t) {
  Vector3<Scalar> v0 = p1 - p0;
  Vector3<Scalar> v1 = p3 - p2;
  Vector3<Scalar> v2 = p2 - p0;
  Scalar a = v0.dot(v0);
  Scalar b = v0.dot(v1);
  Scalar c = v0.dot(v2);
  Scalar d = v1.dot(v1);
  Scalar e = v1.dot(v2);
  Scalar f = v2.dot(v2);
  Scalar det = a * d - b * b;
  Scalar s = b * e - c * d;
  Scalar t_ = b * c - a * e;
  Scalar inv_det = 1 / det;
  if (det > 0) {
    if (s >= 0 && s <= det && t_ >= 0 && t_ <= det) {
      *t = s * inv_det;
      return true;
    }
  } else {
    if (s <= 0 && s >= det && t_ <= 0 && t_ >= det) {
      *t = s * inv_det;
      return true;
    }
  }
  return false;
}

template <typename Scalar>
LM_DEVICE_FUNC bool EdgeEdgeCCD(const Vector3<Scalar> &p0,
                                const Vector3<Scalar> &p1,
                                const Vector3<Scalar> &v0,
                                const Vector3<Scalar> &v1,
                                const Vector3<Scalar> &p2,
                                const Vector3<Scalar> &p3,
                                const Vector3<Scalar> &v2,
                                const Vector3<Scalar> &v3,
                                Scalar *t) {
  Scalar polynomial_terms[4];
  Scalar roots[3];
  ThirdOrderVolumetricPolynomial(p1 - p0, p2 - p0, p3 - p0, v1 - v0, v2 - v0,
                                 v3 - v0, polynomial_terms);
  int num_roots = 0;
  SolveCubicPolynomial(polynomial_terms[3], polynomial_terms[2],
                       polynomial_terms[1], polynomial_terms[0], roots,
                       &num_roots);
  PrivateSort(roots, num_roots);
  for (int i = 0; i < num_roots; i++) {
    if (roots[i] >= 0) {
      *t = roots[i];
      return true;
    }
  }
  return false;
}

}  // namespace grassland::geometry