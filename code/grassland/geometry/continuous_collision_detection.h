#pragma once
#include "grassland/geometry/geometry_util.h"

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
                                         const Vector3<Scalar> &p3) {
  Vector3<Scalar> e1 = p1 - p0;
  Vector3<Scalar> e2 = p3 - p2;
  Vector3<Scalar> normal = e1.cross(e2);
  if (normal.norm() < Eps<Scalar>()) {
    return false;
  }
  normal.normalize();
  e1 = e1.cross(normal);
  e2 = e2.cross(normal);
  e1.normalize();
  e2.normalize();
  Scalar d1 = e1.dot(p0);
  Scalar d2 = e2.dot(p2);
  int sig_product_e1 = Sign(e1.dot(p2) - d1) * Sign(e1.dot(p3) - d1);
  int sig_product_e2 = Sign(e2.dot(p0) - d2) * Sign(e2.dot(p1) - d2);
  return sig_product_e1 <= 0 && sig_product_e2 <= 0;
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
  ThirdOrderVolumetricPolynomial<Scalar>(p1 - p0, p2 - p0, p3 - p0, v1 - v0,
                                         v2 - v0, v3 - v0, polynomial_terms);
  int num_roots = 0;
  SolveCubicPolynomialLimitedRange(polynomial_terms[3], polynomial_terms[2],
                                   polynomial_terms[1], polynomial_terms[0],
                                   roots, &num_roots);
  PrivateSort(roots, num_roots);
  for (int i = 0; i < num_roots; i++) {
    auto root = roots[i];
    if (root > *t) {
      break;
    }
    if (root >= 0) {
      if (EdgeEdgeIntersection<Scalar>(p0 + v0 * root, p1 + v1 * root,
                                       p2 + v2 * root, p3 + v3 * root)) {
        *t = root;
        return true;
      }
    }
  }
  return false;
}

template <typename Scalar>
LM_DEVICE_FUNC bool FacePointIntersection(const Vector3<Scalar> &p0,
                                          const Vector3<Scalar> &p1,
                                          const Vector3<Scalar> &p2,
                                          const Vector3<Scalar> &p) {
  Vector3<Scalar> e0 = p2 - p1;
  Vector3<Scalar> e1 = p0 - p2;
  Vector3<Scalar> e2 = p1 - p0;
  Vector3<Scalar> n = e1.cross(e2);
  if (n.norm() < Eps<Scalar>()) {
    return false;
  }
  n.normalize();
  e0 = e0.cross(n);
  e0.normalize();
  e1 = e1.cross(n);
  e1.normalize();
  e2 = e2.cross(n);
  e2.normalize();
  Scalar d0 = e0.dot(p2);
  Scalar d1 = e1.dot(p0);
  Scalar d2 = e2.dot(p1);
  int sig_product_e0 = -Sign(e0.dot(p) - d0);
  int sig_product_e1 = -Sign(e1.dot(p) - d1);
  int sig_product_e2 = -Sign(e2.dot(p) - d2);
  return sig_product_e0 >= 0 && sig_product_e1 >= 0 && sig_product_e2 >= 0;
}

template <typename Scalar>
LM_DEVICE_FUNC bool FacePointCCD(const Vector3<Scalar> &p0,
                                 const Vector3<Scalar> &p1,
                                 const Vector3<Scalar> &p2,
                                 const Vector3<Scalar> &v0,
                                 const Vector3<Scalar> &v1,
                                 const Vector3<Scalar> &v2,
                                 const Vector3<Scalar> &p,
                                 const Vector3<Scalar> &v,
                                 Scalar *t) {
  Scalar polynomial_terms[4];
  Scalar roots[3];
  ThirdOrderVolumetricPolynomial<Scalar>(p0 - p, p1 - p, p2 - p, v0 - v, v1 - v,
                                         v2 - v, polynomial_terms);
  int num_roots = 0;
  SolveCubicPolynomialLimitedRange(polynomial_terms[3], polynomial_terms[2],
                                   polynomial_terms[1], polynomial_terms[0],
                                   roots, &num_roots);
  PrivateSort(roots, num_roots);
  for (int i = 0; i < num_roots; i++) {
    auto root = roots[i];
    if (root > *t) {
      break;
    }
    if (root >= 0) {
      if (FacePointIntersection<Scalar>(p0 + v0 * root, p1 + v1 * root,
                                        p2 + v2 * root, p + v * root)) {
        *t = root;
        return true;
      }
    }
  }
  return false;
}

}  // namespace grassland::geometry
