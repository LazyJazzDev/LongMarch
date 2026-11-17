#pragma once
#include <cmath>

#include "aero_model.h"

#ifdef __CUDACC__
#define WIND3D_GEOMETRY_HD __host__ __device__
#else
#define WIND3D_GEOMETRY_HD
#endif

namespace wind3d::aero {
namespace mesh_math {
WIND3D_GEOMETRY_HD inline Vector Add(Vector a, Vector b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

WIND3D_GEOMETRY_HD inline Vector Sub(Vector a, Vector b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

WIND3D_GEOMETRY_HD inline Vector Mul(Vector a, double s) {
  return {a.x * s, a.y * s, a.z * s};
}

WIND3D_GEOMETRY_HD inline double Dot(Vector a, Vector b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

WIND3D_GEOMETRY_HD inline Vector Cross(Vector a, Vector b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
}  // namespace mesh_math

struct MeshTriangle {
  Vector vertex[3], vertex_normal[3], edge_normal[3], normal;
};

struct MeshNode {
  Vector lower, upper;
  int triangle{-1}, child{-1}, next{-1};
};

struct MeshHit {
  double squared_distance{1e300};
  Vector point{}, normal{};
  int triangle{-1};
};

// Closest feature carries its angle-weighted pseudonormal, including at creases.
WIND3D_GEOMETRY_HD inline void TriangleHit(const MeshTriangle &t, Vector p, int index, MeshHit &hit) {
  using namespace mesh_math;
  Vector a = t.vertex[0], b = t.vertex[1], c = t.vertex[2], ab = Sub(b, a), ac = Sub(c, a), ap = Sub(p, a);
  Vector point, normal;
  double d1 = Dot(ab, ap), d2 = Dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) {
    point = a;
    normal = t.vertex_normal[0];
  } else {
    Vector bp = Sub(p, b);
    double d3 = Dot(ab, bp), d4 = Dot(ac, bp), vc = d1 * d4 - d3 * d2;
    if (d3 >= 0 && d4 <= d3) {
      point = b;
      normal = t.vertex_normal[1];
    } else if (vc <= 0 && d1 >= 0 && d3 <= 0) {
      point = Add(a, Mul(ab, d1 / (d1 - d3)));
      normal = t.edge_normal[0];
    } else {
      Vector cp = Sub(p, c);
      double d5 = Dot(ab, cp), d6 = Dot(ac, cp), vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
      if (d6 >= 0 && d5 <= d6) {
        point = c;
        normal = t.vertex_normal[2];
      } else if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        point = Add(a, Mul(ac, d2 / (d2 - d6)));
        normal = t.edge_normal[2];
      } else if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
        point = Add(b, Mul(Sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
        normal = t.edge_normal[1];
      } else {
        double inv = 1 / (va + vb + vc);
        point = Add(a, Add(Mul(ab, vb * inv), Mul(ac, vc * inv)));
        normal = t.normal;
      }
    }
  }
  double distance = Dot(Sub(p, point), Sub(p, point));
  if (distance < hit.squared_distance)
    hit = {distance, point, normal, index};
}

struct MeshView {
  const MeshTriangle *triangles{};
  const MeshNode *nodes{};
  int node_count{};

  WIND3D_GEOMETRY_HD MeshHit Closest(Vector p) const {
    MeshHit hit;
    int n = node_count ? 0 : -1;
    while (n >= 0) {
      const auto &node = nodes[n];
      double x = fmax(fmax(node.lower.x - p.x, p.x - node.upper.x), 0.0);
      double y = fmax(fmax(node.lower.y - p.y, p.y - node.upper.y), 0.0);
      double z = fmax(fmax(node.lower.z - p.z, p.z - node.upper.z), 0.0);
      if (x * x + y * y + z * z > hit.squared_distance)
        n = node.next;
      else if (node.triangle >= 0) {
        TriangleHit(triangles[node.triangle], p, node.triangle, hit);
        n = node.next;
      } else
        n = node.child;
    }
    return hit;
  }

  WIND3D_GEOMETRY_HD double Distance(Vector p) const {
    auto hit = Closest(p);
    double d = sqrt(hit.squared_distance);
    return mesh_math::Dot(mesh_math::Sub(p, hit.point), hit.normal) < 0 ? -d : d;
  }
};
}  // namespace wind3d::aero

#undef WIND3D_GEOMETRY_HD
