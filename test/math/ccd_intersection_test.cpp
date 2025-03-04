#include "gtest/gtest.h"
#include "long_march.h"
#include "random"
#include "vector"

using namespace long_march;

template <typename Scalar>
bool RandomEdgeEdge(Vector3<Scalar> *edges) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Scalar> dis(-1, 1);
  Vector2<Scalar> _2d_points[4];
  do {
    for (int i = 0; i < 4; i++) {
      _2d_points[i] = Vector2<Scalar>(dis(gen), dis(gen));
    }
  } while (fabs(PolygonArea(_2d_points, 4)) < Eps<Scalar>());
  bool pos_sig = false, neg_sig = false;
  for (int i = 0; i < 4; i++) {
    int i_1 = (i + 1) % 4;
    int i_2 = (i + 2) % 4;
    Vector2<Scalar> p0 = _2d_points[i];
    Vector2<Scalar> p1 = _2d_points[i_1];
    Vector2<Scalar> p2 = _2d_points[i_2];
    Vector2<Scalar> e0 = p0 - p1;
    Vector2<Scalar> e1 = p2 - p1;
    Scalar cross = e0[0] * e1[1] - e0[1] * e1[0];
    if (cross > 0) {
      pos_sig = true;
    } else if (cross < 0) {
      neg_sig = true;
    }
  }
  Vector3<Scalar> random_axis_x;
  Vector3<Scalar> random_axis_y;
  do {
    random_axis_x = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    random_axis_y = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
  } while (random_axis_x.cross(random_axis_y).norm() < 0.1);
  Matrix<Scalar, 3, 2> random_axes;
  random_axes.col(0) = random_axis_x;
  random_axes.col(1) = random_axis_y;
  edges[0] = random_axes * _2d_points[0];
  edges[1] = random_axes * _2d_points[2];
  edges[2] = random_axes * _2d_points[1];
  edges[3] = random_axes * _2d_points[3];
  return !(pos_sig && neg_sig);
}

template <typename Scalar>
void TestEdgeEdgeIntersection() {
  Vector3<Scalar> edges[4];
  bool answer = RandomEdgeEdge<Scalar>(edges);
  EXPECT_EQ(EdgeEdgeIntersection(edges[0], edges[1], edges[2], edges[3]), answer);
}

template <typename Scalar>
bool RandomFacePoint(Vector3<Scalar> *vs) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Scalar> dis(-1, 1);
  Vector3<Scalar> v_base;
  Vector3<Scalar> v0;
  Vector3<Scalar> v1;
  v_base = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
  do {
    do {
      v0 = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    } while (v0.norm() < 0.1);
    do {
      v1 = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    } while (v1.norm() < 0.1);
  } while (v0.cross(v1).norm() < Eps<Scalar>());
  Vector2<Scalar> barycentric(dis(gen), dis(gen));
  barycentric = barycentric * 1.5 + Vector2<Scalar>(0.5, 0.5);
  Matrix<Scalar, 3, 2> m;
  m.col(0) = v0;
  m.col(1) = v1;
  vs[0] = v_base;
  vs[1] = v_base + v0;
  vs[2] = v_base + v1;
  vs[3] = v_base + m * barycentric;
  return barycentric.sum() <= 1 && barycentric[0] >= 0 && barycentric[1] >= 0;
}

template <typename Scalar>
void TestFacePointIntersection() {
  Vector3<Scalar> vs[4];
  bool answer = RandomFacePoint<Scalar>(vs);
  EXPECT_EQ(FacePointIntersection(vs[0], vs[1], vs[2], vs[3]), answer);
}

template <typename Scalar>
void TestEdgeEdgeIntersectionVolumeSolve() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Scalar> dis(-1, 1);
  Vector3<Scalar> p0, p1, p2, p3;
  Vector3<Scalar> v0, v1, v2, v3;
  do {
    p0 = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    p1 = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    p2 = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    p3 = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    v0 = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    v1 = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    v2 = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    v3 = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
  } while (fabs(TetrahedronVolume(p0, p1, p2, p3)) < Eps<Scalar>() || fabs(TetrahedronVolume(v0, v1, v2, v3)) < 1e-2);
  Scalar t = 1;
  if (EdgeEdgeCCD(p0, p1, v0, v1, p2, p3, v2, v3, &t)) {
    Vector3<Scalar> x0 = p0 + v0 * t;
    Vector3<Scalar> x1 = p1 + v1 * t;
    Vector3<Scalar> x2 = p2 + v2 * t;
    Vector3<Scalar> x3 = p3 + v3 * t;
    EXPECT_TRUE(EdgeEdgeIntersection(x0, x1, x2, x3));
    EXPECT_NEAR(TetrahedronVolume(x0, x1, x2, x3), 0, Eps<Scalar>() * fmax(t, static_cast<Scalar>(1)));
    // If expect is not satisfied, print the result
    //        if (fabs(TetrahedronVolume(x0, x1, x2, x3)) >
    //            Eps<Scalar>() * fmax(t, static_cast<Scalar>(1))) {
    //            Scalar poly[4]{};
    //            ThirdOrderVolumetricPolynomial<Scalar>(
    //                    p1 - p0, p2 - p0, p3 - p0, v1 - v0, v2 - v0, v3 - v0,
    //                    poly);
    //            Scalar roots[3];
    //            int num_roots;
    //            SolveCubicPolynomial(poly[3], poly[2], poly[1],
    //            poly[0], roots, &num_roots); std::cout << "Roots:\n"; for (int
    //            num_root = 0; num_root < num_roots; num_root++) {
    //                Scalar x = roots[num_root];
    //                std::cout << x << ":";
    //                std::cout << poly[3] * x * x * x + poly[2] * x * x +
    //                poly[1] * x + poly[0]
    //                          << "\n";
    //            }
    //            std::cout << "Poly solution: "
    //                      << poly[3] * t * t * t + poly[2] * t * t + poly[1] *
    //                      t + poly[0]
    //                      << std::endl;
    //            std::cout << "Poly: " << poly[3] << "x^3 + " << poly[2] <<
    //            "x^2 + "
    //                      << poly[1] << "x + " << poly[0] << std::endl;
    //            std::cout << "Time: " << t << std::endl;
    //            std::cout << "Before Volume: "
    //                      << TetrahedronVolume(p0, p1, p2, p3) * 6
    //                      << std::endl;
    //            std::cout << "After Volume: "
    //                      << TetrahedronVolume(x0, x1, x2, x3) * 6
    //                      << std::endl;
    //            std::cout << "Velocity Volume: "
    //                      << TetrahedronVolume<Scalar>(v0, v1, v2,
    //                      v3) * 6
    //                      << std::endl;
    //        }
  }
}

template <typename Scalar>
void TestFacePointIntersectionVolumeSolve() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Scalar> dis(-1, 1);
  Vector3<Scalar> p0, p1, p2, p3;
  Vector3<Scalar> v0, v1, v2, v3;
  do {
    v0 = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    v1 = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    v2 = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    v3 = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    p0 = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    p1 = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    p2 = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    p3 = Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
  } while (TetrahedronVolume(p0, p1, p2, p3) < Eps<Scalar>() || TetrahedronVolume(v0, v1, v2, v3) < 1e-2);
  Scalar t = 1;
  if (FacePointCCD(p0, p1, p2, v0, v1, v2, p3, v3, &t)) {
    Vector3<Scalar> x0 = p0 + v0 * t;
    Vector3<Scalar> x1 = p1 + v1 * t;
    Vector3<Scalar> x2 = p2 + v2 * t;
    Vector3<Scalar> x3 = p3 + v3 * t;
    EXPECT_TRUE(FacePointIntersection(x0, x1, x2, x3));
    EXPECT_NEAR(TetrahedronVolume(x0, x1, x2, x3), 0, Eps<Scalar>() * fmax(t, static_cast<Scalar>(1)));
  }
}

template <typename Scalar>
void BatchedTest() {
  for (int i = 0; i < 100000; i++) {
    TestEdgeEdgeIntersection<Scalar>();
    TestFacePointIntersection<Scalar>();
    TestEdgeEdgeIntersectionVolumeSolve<Scalar>();
    TestFacePointIntersectionVolumeSolve<Scalar>();
  }
}

TEST(Math, CCDIntersection) {
  BatchedTest<float>();
  BatchedTest<double>();
}
