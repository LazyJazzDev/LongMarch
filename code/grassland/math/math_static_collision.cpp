#include "grassland/math/math_static_collision.h"

namespace grassland {

template <class Real>
LM_DEVICE_FUNC Real DistancePointPoint(const Vector3<Real> &p0, const Vector3<Real> &p1) {
  return (p0 - p1).norm();
}

template LM_DEVICE_FUNC float DistancePointPoint(const Vector3<float> &p0, const Vector3<float> &p1);
template LM_DEVICE_FUNC double DistancePointPoint(const Vector3<double> &p0, const Vector3<double> &p1);

template <class Real>
LM_DEVICE_FUNC Real
DistancePointLine(const Vector3<Real> &p, const Vector3<Real> &s0, const Vector3<Real> &s1, Real &t) {
  Vector3<Real> ds = s1 - s0;
  // d(t) = p - (s0 + t * ds) = (p - s0) - t * ds
  // E = d^2 = (p - s0)^2 - 2 * t * (p - s0) dot ds + t^2 * ds^2
  // dE/dt = -2 * (p - s0) dot ds + 2 * t * ds^2 = 0
  // t = (p - s0) dot ds / ds^2
  t = (p - s0).dot(ds) / ds.squaredNorm();
  return (p - s0 - t * ds).norm();
}

template LM_DEVICE_FUNC float DistancePointLine(const Vector3<float> &p,
                                                const Vector3<float> &s0,
                                                const Vector3<float> &s1,
                                                float &t);
template LM_DEVICE_FUNC double DistancePointLine(const Vector3<double> &p,
                                                 const Vector3<double> &s0,
                                                 const Vector3<double> &s1,
                                                 double &t);

template <class Real>
LM_DEVICE_FUNC Real DistancePointLine(const Vector3<Real> &p, const Vector3<Real> &s0, const Vector3<Real> &s1) {
  Real t;
  return DistancePointLine(p, s0, s1, t);
}

template LM_DEVICE_FUNC float DistancePointLine(const Vector3<float> &p,
                                                const Vector3<float> &s0,
                                                const Vector3<float> &s1);
template LM_DEVICE_FUNC double DistancePointLine(const Vector3<double> &p,
                                                 const Vector3<double> &s0,
                                                 const Vector3<double> &s1);

template <class Real>
LM_DEVICE_FUNC Real
DistancePointSegment(const Vector3<Real> &p, const Vector3<Real> &s0, const Vector3<Real> &s1, Real &t) {
  Vector3<Real> ds = s1 - s0;
  // d(t) = p - (s0 + t * ds) = (p - s0) - t * ds
  // E = d^2 = (p - s0)^2 - 2 * t * (p - s0) dot ds + t^2 * ds^2
  // dE/dt = -2 * (p - s0) dot ds + 2 * t * ds^2 = 0
  // t = (p - s0) dot ds / ds^2
  t = (p - s0).dot(ds) / ds.squaredNorm();
  t = fmax(0, fmin(1, t));
  return (p - s0 - t * ds).norm();
}

template LM_DEVICE_FUNC float DistancePointSegment(const Vector3<float> &p,
                                                   const Vector3<float> &s0,
                                                   const Vector3<float> &s1,
                                                   float &t);
template LM_DEVICE_FUNC double DistancePointSegment(const Vector3<double> &p,
                                                    const Vector3<double> &s0,
                                                    const Vector3<double> &s1,
                                                    double &t);

template <class Real>
LM_DEVICE_FUNC Real DistancePointSegment(const Vector3<Real> &p, const Vector3<Real> &s0, const Vector3<Real> &s1) {
  Real t;
  return DistancePointSegment(p, s0, s1, t);
}

template LM_DEVICE_FUNC float DistancePointSegment(const Vector3<float> &p,
                                                   const Vector3<float> &s0,
                                                   const Vector3<float> &s1);
template LM_DEVICE_FUNC double DistancePointSegment(const Vector3<double> &p,
                                                    const Vector3<double> &s0,
                                                    const Vector3<double> &s1);

template <class Real>
LM_DEVICE_FUNC Real DistancePointPlane(const Vector3<Real> &p,
                                       const Vector3<Real> &v0,
                                       const Vector3<Real> &v1,
                                       const Vector3<Real> &v2,
                                       Real &u,
                                       Real &v) {
  Vector3<Real> ds0 = v1 - v0;
  Vector3<Real> ds1 = v2 - v0;
  Vector3<Real> dp = p - v0;
  // clang-format off
  // d(u, v) = p - v0 - u * ds0 - v * ds1
  // E = d^2 = dp^2 - 2 * u * dp dot ds0 - 2 * v * dp dot ds1 + 2 * uv * ds0 dot ds1 + u^2 * ds0^2 + v^2 * ds1^2
  // = a + 2bu + 2cv + du^2 + ev^2 + 2fuv
  // dE/du = 2b + 2u * d + 2v * f = 0
  // dE/dv = 2c + 2v * e + 2u * f = 0
  // 2d * u + 2f * v = -2b
  // 2f * u + 2e * v = -2c
  // du + fv = -b
  // fu + ev = -c
  // clang-format on
  Real b = dp.dot(ds0);
  Real c = dp.dot(ds1);
  Real d = ds0.squaredNorm();
  Real e = ds1.squaredNorm();
  Real f = ds0.dot(ds1);
  // Ax = b
  Matrix<Real, 2, 2> A;
  A << d, f, f, e;
  A = A.inverse();
  u = A(0, 0) * b + A(0, 1) * c;
  v = A(1, 0) * b + A(1, 1) * c;
  return (dp - u * ds0 - v * ds1).norm();
}

template LM_DEVICE_FUNC float DistancePointPlane(const Vector3<float> &p,
                                                 const Vector3<float> &v0,
                                                 const Vector3<float> &v1,
                                                 const Vector3<float> &v2,
                                                 float &u,
                                                 float &v);
template LM_DEVICE_FUNC double DistancePointPlane(const Vector3<double> &p,
                                                  const Vector3<double> &v0,
                                                  const Vector3<double> &v1,
                                                  const Vector3<double> &v2,
                                                  double &u,
                                                  double &v);

template <class Real>
LM_DEVICE_FUNC Real
DistancePointPlane(const Vector3<Real> &p, const Vector3<Real> &v0, const Vector3<Real> &v1, const Vector3<Real> &v2) {
  Vector3<Real> normal = (v1 - v0).cross(v2 - v0);
  normal.normalize();
  return fabs(normal.dot(p - v0));
}

template LM_DEVICE_FUNC float DistancePointPlane(const Vector3<float> &p,
                                                 const Vector3<float> &v0,
                                                 const Vector3<float> &v1,
                                                 const Vector3<float> &v2);
template LM_DEVICE_FUNC double DistancePointPlane(const Vector3<double> &p,
                                                  const Vector3<double> &v0,
                                                  const Vector3<double> &v1,
                                                  const Vector3<double> &v2);

template <class Real>
LM_DEVICE_FUNC Real DistancePointTriangleProjection(const Vector3<Real> &p,
                                                    const Vector3<Real> &v0,
                                                    const Vector3<Real> &v1,
                                                    const Vector3<Real> &v2) {
  Real u, v;
  Real t = DistancePointPlane(p, v0, v1, v2, u, v);
  if (u >= 0 && v >= 0 && u + v <= 1) {
    return t;
  }
  return -1;
}

template LM_DEVICE_FUNC float DistancePointTriangleProjection(const Vector3<float> &p,
                                                              const Vector3<float> &v0,
                                                              const Vector3<float> &v1,
                                                              const Vector3<float> &v2);
template LM_DEVICE_FUNC double DistancePointTriangleProjection(const Vector3<double> &p,
                                                               const Vector3<double> &v0,
                                                               const Vector3<double> &v1,
                                                               const Vector3<double> &v2);

template <class Real>
LM_DEVICE_FUNC Real DistancePointTriangle(const Vector3<Real> &p,
                                          const Vector3<Real> &v0,
                                          const Vector3<Real> &v1,
                                          const Vector3<Real> &v2,
                                          Real &u,
                                          Real &v) {
  Real t = DistancePointPlane(p, v0, v1, v2, u, v);
  if (u >= 0 && v >= 0 && u + v <= 1) {
    return t;
  }
  t = DistancePointSegment(p, v0, v1, u);
  v = 0;
  Real w;
  Real t1 = DistancePointSegment(p, v1, v2, w);
  if (t1 < t) {
    t = t1;
    u = 1 - w;
    v = w;
  }
  t1 = DistancePointSegment(p, v0, v2, w);
  if (t1 < t) {
    t = t1;
    u = 0;
    v = w;
  }
  return t;
}

template LM_DEVICE_FUNC float DistancePointTriangle(const Vector3<float> &p,
                                                    const Vector3<float> &v0,
                                                    const Vector3<float> &v1,
                                                    const Vector3<float> &v2,
                                                    float &u,
                                                    float &v);
template LM_DEVICE_FUNC double DistancePointTriangle(const Vector3<double> &p,
                                                     const Vector3<double> &v0,
                                                     const Vector3<double> &v1,
                                                     const Vector3<double> &v2,
                                                     double &u,
                                                     double &v);

template <class Real>
LM_DEVICE_FUNC Real DistancePointTriangle(const Vector3<Real> &p,
                                          const Vector3<Real> &v0,
                                          const Vector3<Real> &v1,
                                          const Vector3<Real> &v2) {
  Real u, v;
  return DistancePointTriangle(p, v0, v1, v2, u, v);
}

template LM_DEVICE_FUNC float DistancePointTriangle(const Vector3<float> &p,
                                                    const Vector3<float> &v0,
                                                    const Vector3<float> &v1,
                                                    const Vector3<float> &v2);
template LM_DEVICE_FUNC double DistancePointTriangle(const Vector3<double> &p,
                                                     const Vector3<double> &v0,
                                                     const Vector3<double> &v1,
                                                     const Vector3<double> &v2);

template <class Real>
LM_DEVICE_FUNC Real DistanceLineLine(const Vector3<Real> &p0,
                                     const Vector3<Real> &p1,
                                     const Vector3<Real> &q0,
                                     const Vector3<Real> &q1,
                                     Real &u,
                                     Real &v) {
  Vector3<Real> pq = p0 - q0;
  Vector3<Real> dp = p1 - p0;
  Vector3<Real> dq = q1 - q0;
  // clang-format off
  // d(u, v) = (p0 + u * dp) - (q0 + v * dq) = pq + u * dp - v * dq
  // E = d^2 = pq^2 + 2u * pq dot dp - 2v * pq dot dq + u^2 * dp^2 - 2uv * dp dot dq + v^2 * dq^2
  // = a + 2bu + 2cv + du^2 + ev^2 + 2fuv
  // dE/du = 2b + 2du + 2fv = 0
  // dE/dv = 2c + 2fu + 2ev = 0
  // clang-format on
  Real a = pq.squaredNorm();
  Real b = dp.dot(pq);
  Real c = -dq.dot(pq);
  Real d = dp.squaredNorm();
  Real e = dq.squaredNorm();
  Real f = -dp.dot(dq);
  // Ax = b
  Matrix2<Real> A;
  A << d, f, f, e;
  if (A.determinant() < Eps<Real>() * Eps<Real>()) {
    u = 0;
    return DistancePointLine(p0, q0, q1, v);
  }
  A = -A.inverse();
  u = A(0, 0) * b + A(0, 1) * c;
  v = A(1, 0) * b + A(1, 1) * c;
  return (pq + u * dp - v * dq).norm();
}

template LM_DEVICE_FUNC float DistanceLineLine(const Vector3<float> &p0,
                                               const Vector3<float> &p1,
                                               const Vector3<float> &q0,
                                               const Vector3<float> &q1,
                                               float &u,
                                               float &v);
template LM_DEVICE_FUNC double DistanceLineLine(const Vector3<double> &p0,
                                                const Vector3<double> &p1,
                                                const Vector3<double> &q0,
                                                const Vector3<double> &q1,
                                                double &u,
                                                double &v);

template <class Real>
LM_DEVICE_FUNC Real
DistanceLineLine(const Vector3<Real> &p0, const Vector3<Real> &p1, const Vector3<Real> &q0, const Vector3<Real> &q1) {
  Real u, v;
  return DistanceLineLine(p0, p1, q0, q1, u, v);
}

template LM_DEVICE_FUNC float DistanceLineLine(const Vector3<float> &p0,
                                               const Vector3<float> &p1,
                                               const Vector3<float> &q0,
                                               const Vector3<float> &q1);
template LM_DEVICE_FUNC double DistanceLineLine(const Vector3<double> &p0,
                                                const Vector3<double> &p1,
                                                const Vector3<double> &q0,
                                                const Vector3<double> &q1);

template <class Real>
LM_DEVICE_FUNC Real DistanceSegmentSegment(const Vector3<Real> &p0,
                                           const Vector3<Real> &p1,
                                           const Vector3<Real> &q0,
                                           const Vector3<Real> &q1,
                                           Real &u,
                                           Real &v) {
  Real t = DistanceLineLine(p0, p1, q0, q1, u, v);
  Real w;
  if (u < 0 || u > 1) {
    if (u < 0) {
      u = 0;
    } else {
      u = 1;
    }
    w = v;
    t = DistancePointLine(Vector3<Real>(p0 + u * (p1 - p0)), q0, q1, v);
    if (!((v < 0 && w < 0) || (v > 1 && w > 1))) {
      v = fmax(0, fmin(1, v));
      t = (p0 + u * (p1 - p0) - q0 - v * (q1 - q0)).norm();
      return t;
    }
  }
  if (v < 0 || v > 1) {
    if (v < 0) {
      v = 0;
    } else {
      v = 1;
    }
    t = DistancePointSegment(Vector3<Real>(q0 + v * (q1 - q0)), p0, p1, u);
  }
  return t;
}

template LM_DEVICE_FUNC float DistanceSegmentSegment(const Vector3<float> &p0,
                                                     const Vector3<float> &p1,
                                                     const Vector3<float> &q0,
                                                     const Vector3<float> &q1,
                                                     float &u,
                                                     float &v);
template LM_DEVICE_FUNC double DistanceSegmentSegment(const Vector3<double> &p0,
                                                      const Vector3<double> &p1,
                                                      const Vector3<double> &q0,
                                                      const Vector3<double> &q1,
                                                      double &u,
                                                      double &v);

template <class Real>
LM_DEVICE_FUNC Real DistanceSegmentSegment(const Vector3<Real> &p0,
                                           const Vector3<Real> &p1,
                                           const Vector3<Real> &q0,
                                           const Vector3<Real> &q1) {
  Real u, v;
  return DistanceSegmentSegment(p0, p1, q0, q1, u, v);
}

template LM_DEVICE_FUNC float DistanceSegmentSegment(const Vector3<float> &p0,
                                                     const Vector3<float> &p1,
                                                     const Vector3<float> &q0,
                                                     const Vector3<float> &q1);
template LM_DEVICE_FUNC double DistanceSegmentSegment(const Vector3<double> &p0,
                                                      const Vector3<double> &p1,
                                                      const Vector3<double> &q0,
                                                      const Vector3<double> &q1);

}  // namespace grassland
