#include "gtest/gtest.h"
#include "long_march.h"
#include "random"
#include "vector"

using namespace long_march;

template <class Real>
void DistancePointPlaneCorrectnessTest() {
  Vector3<Real> p = Vector3<Real>::Random();
  Vector3<Real> v0, v1, v2;
  do {
    v0 = Vector3<Real>::Random();
    v1 = Vector3<Real>::Random();
    v2 = Vector3<Real>::Random();
  } while ((v1 - v0).cross(v2 - v0).norm() < 0.1);
  Real u, v;
  Real d = DistancePointPlane(p, v0, v1, v2, u, v);
  Real d2 = DistancePointPlane(p, v0, v1, v2);
  Vector3<Real> p_proj = v0 + u * (v1 - v0) + v * (v2 - v0);
  EXPECT_NEAR((p - p_proj).cross((v1 - v0).cross(v2 - v0)).norm(), 0, Eps<Real>());
  EXPECT_NEAR(d, d2, Eps<Real>());
}

TEST(Math, DistancePointPlaneCorrectness) {
  for (int i = 0; i < 10000; i++) {
    DistancePointPlaneCorrectnessTest<float>();
  }
  for (int i = 0; i < 10000; i++) {
    DistancePointPlaneCorrectnessTest<double>();
  }
}

template <class Real>
void DistancePointTriangleCorrectnessTest() {
  Vector3<Real> p = Vector3<Real>::Random();
  Vector3<Real> v0, v1, v2;
  do {
    v0 = Vector3<Real>::Random();
    v1 = Vector3<Real>::Random();
    v2 = Vector3<Real>::Random();
  } while ((v1 - v0).cross(v2 - v0).norm() < Eps<Real>());
  Real u, v;
  Real d = DistancePointTriangle(p, v0, v1, v2, u, v);
  Vector3<Real> p_proj = v0 + u * (v1 - v0) + v * (v2 - v0);
  EXPECT_NEAR((p_proj - p).norm(), d, Eps<Real>());
}

TEST(Math, DistancePointTriangleCorrectness) {
  for (int i = 0; i < 10000; i++) {
    DistancePointTriangleCorrectnessTest<float>();
  }
  for (int i = 0; i < 10000; i++) {
    DistancePointTriangleCorrectnessTest<double>();
  }
}

template <class Real>
void DistanceLineLineCorrectnessTest() {
  Vector3<Real> p0, p1;
  Vector3<Real> q0, q1;
  do {
    p0 = Vector3<Real>::Random();
    p1 = Vector3<Real>::Random();
    q0 = Vector3<Real>::Random();
    q1 = Vector3<Real>::Random();
  } while ((p1 - p0).cross(q1 - q0).norm() < Eps<Real>());
  Real u, v;
  Real d = DistanceLineLine(p0, p1, q0, q1, u, v);
  Vector3<Real> p = p0 + u * (p1 - p0);
  Vector3<Real> q = q0 + v * (q1 - q0);
  EXPECT_NEAR((p - q).cross((q1 - q0).cross(p1 - p0)).norm(), 0, Eps<Real>());
  EXPECT_NEAR(d, (p - q).norm(), Eps<Real>());
}

TEST(Math, DistanceLineLineCorrectness) {
  for (int i = 0; i < 10000; i++) {
    DistanceLineLineCorrectnessTest<float>();
  }
  for (int i = 0; i < 10000; i++) {
    DistanceLineLineCorrectnessTest<double>();
  }
}

template <class Real>
void DistanceSegmentSegmentCorrectnessTest() {
  Vector3<Real> p0, p1;
  Vector3<Real> q0, q1;

  p0 = Vector3<Real>::Random();
  p1 = Vector3<Real>::Random();
  q0 = Vector3<Real>::Random();
  q1 = Vector3<Real>::Random();

  Real u, v;
  Real d = DistanceSegmentSegment(p0, p1, q0, q1, u, v);
  Vector3<Real> p = p0 + u * (p1 - p0);
  Vector3<Real> q = q0 + v * (q1 - q0);
  EXPECT_NEAR(d, (p - q).norm(), Eps<Real>());
  d = (p - q).norm();
  std::set<Real> us;
  std::set<Real> vs;
  for (int i = 0; i < 10; i++) {
    us.insert((Eigen::Matrix<Real, 1, 1>::Random().value() + 1) / 2);
    if (u + i * Eps<Real>() < 1) {
      us.insert(u + i * Eps<Real>());
    }
    if (u - i * Eps<Real>() > 0) {
      us.insert(u - i * Eps<Real>());
    }
  }
  for (int i = 0; i < 10; i++) {
    vs.insert((Eigen::Matrix<Real, 1, 1>::Random().value() + 1) / 2);
    if (v + i * Eps<Real>() < 1) {
      vs.insert(v + i * Eps<Real>());
    }
    if (v - i * Eps<Real>() > 0) {
      vs.insert(v - i * Eps<Real>());
    }
  }
  for (Real u : us) {
    for (Real v : vs) {
      Vector3<Real> p = p0 + u * (p1 - p0);
      Vector3<Real> q = q0 + v * (q1 - q0);
      Real d2 = (p - q).norm();
      EXPECT_LT(d, d2 + Eps<Real>()), printf("u: %f, v: %f, d: %f, d2: %f\n", u, v, d, d2);
    }
  }
}

TEST(Math, DistanceSegmentSegmentCorrectness) {
  for (int i = 0; i < 10000; i++) {
    DistanceSegmentSegmentCorrectnessTest<float>();
  }
  for (int i = 0; i < 10000; i++) {
    DistanceSegmentSegmentCorrectnessTest<double>();
  }
}
