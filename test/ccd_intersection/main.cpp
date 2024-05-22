#include "gtest/gtest.h"
#include "long_march.h"
#include "random"
#include "vector"

using namespace long_march;

template <typename Scalar>
bool RandomEdgeEdge(geometry::Vector3<Scalar> *edges) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Scalar> dis(-1, 1);
  geometry::Vector2<Scalar> _2d_points[4];
  do {
    for (int i = 0; i < 4; i++) {
      _2d_points[i] = geometry::Vector2<Scalar>(dis(gen), dis(gen));
    }
  } while (fabs(geometry::PolygonArea(_2d_points, 4)) <
           geometry::Eps<Scalar>());
  bool pos_sig = false, neg_sig = false;
  for (int i = 0; i < 4; i++) {
    int i_1 = (i + 1) % 4;
    int i_2 = (i + 2) % 4;
    geometry::Vector2<Scalar> p0 = _2d_points[i];
    geometry::Vector2<Scalar> p1 = _2d_points[i_1];
    geometry::Vector2<Scalar> p2 = _2d_points[i_2];
    geometry::Vector2<Scalar> e0 = p0 - p1;
    geometry::Vector2<Scalar> e1 = p2 - p1;
    Scalar cross = e0[0] * e1[1] - e0[1] * e1[0];
    if (cross > 0) {
      pos_sig = true;
    } else if (cross < 0) {
      neg_sig = true;
    }
  }
  geometry::Vector3<Scalar> random_axis_x;
  geometry::Vector3<Scalar> random_axis_y;
  do {
    random_axis_x = geometry::Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    random_axis_y = geometry::Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
  } while (random_axis_x.cross(random_axis_y).norm() < 0.1);
  geometry::Matrix<Scalar, 3, 2> random_axes;
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
  geometry::Vector3<Scalar> edges[4];
  bool answer = RandomEdgeEdge<Scalar>(edges);
  EXPECT_EQ(
      geometry::EdgeEdgeIntersection(edges[0], edges[1], edges[2], edges[3]),
      answer);
};

template <typename Scalar>
bool RandomFacePoint(geometry::Vector3<Scalar> *vs) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Scalar> dis(-1, 1);
  geometry::Vector3<Scalar> v_base;
  geometry::Vector3<Scalar> v0;
  geometry::Vector3<Scalar> v1;
  v_base = geometry::Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
  do {
    do {
      v0 = geometry::Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    } while (v0.norm() < 0.1);
    do {
      v1 = geometry::Vector3<Scalar>(dis(gen), dis(gen), dis(gen));
    } while (v1.norm() < 0.1);
  } while (v0.cross(v1).norm() < geometry::Eps<Scalar>());
  geometry::Vector2<Scalar> barycentric(dis(gen), dis(gen));
  barycentric = barycentric * 1.5 + geometry::Vector2<Scalar>(0.5, 0.5);
  geometry::Matrix<Scalar, 3, 2> m;
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
  geometry::Vector3<Scalar> vs[4];
  bool answer = RandomFacePoint<Scalar>(vs);
  EXPECT_EQ(geometry::FacePointIntersection(vs[0], vs[1], vs[2], vs[3]),
            answer);
}

template <typename Scalar>
void BatchedTest() {
  for (int i = 0; i < 100000; i++) {
    TestEdgeEdgeIntersection<Scalar>();
    TestFacePointIntersection<Scalar>();
  }
}

TEST(Geometry, CCDIntersection) {
  BatchedTest<float>();
  BatchedTest<double>();
}
