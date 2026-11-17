#include "aero_mesh.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <unordered_map>

#include "grassland/bvh/bvh_host.h"
#include "tiny_obj_loader.h"

namespace wind3d::aero {
using namespace mesh_math;

std::shared_ptr<const MeshGeometry> MeshGeometry::LoadObj(const std::string &path, std::string &error) {
  error.clear();
  auto fail = [&](const std::string &message) -> std::shared_ptr<const MeshGeometry> {
    error = path + ": " + message;
    return {};
  };
  std::ifstream stream(std::filesystem::u8path(path));
  if (!stream)
    return fail("Cannot open mesh");
  tinyobj::attrib_t attributes;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warning, parse_error;
  if (!tinyobj::LoadObj(&attributes, &shapes, &materials, &warning, &parse_error, &stream, nullptr, true))
    return fail(parse_error);
  std::vector<Vector> vertices;
  std::vector<int> remap;
  std::map<std::array<double, 3>, int> welded;
  for (size_t i = 0; i + 2 < attributes.vertices.size(); i += 3) {
    Vector p{attributes.vertices[i], attributes.vertices[i + 1], attributes.vertices[i + 2]};
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
      return fail("Nonfinite vertex");
    auto result = welded.emplace(std::array<double, 3>{p.x, p.y, p.z}, int(vertices.size()));
    if (result.second)
      vertices.push_back(p);
    remap.push_back(result.first->second);
  }
  std::vector<std::array<int, 3>> indices;
  for (const auto &shape : shapes) {
    size_t offset = 0;
    for (auto count : shape.mesh.num_face_vertices) {
      if (count != 3 || offset + 3 > shape.mesh.indices.size())
        return fail("Triangulation failed");
      std::array<int, 3> face;
      for (int j = 0; j < 3; ++j) {
        int v = shape.mesh.indices[offset + j].vertex_index;
        if (v < 0 || size_t(v) >= remap.size())
          return fail("Invalid vertex index");
        face[j] = remap[v];
      }
      indices.push_back(face);
      offset += 3;
    }
  }
  if (indices.empty() || indices.size() > size_t(std::numeric_limits<int>::max() / 2))
    return fail("Invalid triangle count");
  auto mesh = std::make_shared<MeshGeometry>();
  mesh->triangles_.resize(indices.size());

  struct Edge {
    int count{}, orientation{};
    Vector normal{};
  };

  std::map<std::pair<int, int>, Edge> edges;
  std::vector<Vector> normals(vertices.size());
  double volume = 0;
  Vector origin = vertices[indices[0][0]];
  for (size_t i = 0; i < indices.size(); ++i) {
    auto &t = mesh->triangles_[i];
    for (int j = 0; j < 3; ++j)
      t.vertex[j] = vertices[indices[i][j]];
    Vector cross = Cross(Sub(t.vertex[1], t.vertex[0]), Sub(t.vertex[2], t.vertex[0]));
    double area = std::sqrt(Dot(cross, cross));
    if (!(area > 0) || !std::isfinite(area))
      return fail("Degenerate triangle");
    t.normal = Mul(cross, 1 / area);
    volume += Dot(Sub(t.vertex[0], origin), Cross(Sub(t.vertex[1], origin), Sub(t.vertex[2], origin))) / 6;
    for (int j = 0; j < 3; ++j) {
      int a = indices[i][j], b = indices[i][(j + 1) % 3];
      auto &edge = edges[std::minmax(a, b)];
      ++edge.count;
      edge.orientation += a < b ? 1 : -1;
      edge.normal = Add(edge.normal, t.normal);
      Vector u = Sub(t.vertex[(j + 1) % 3], t.vertex[j]), v = Sub(t.vertex[(j + 2) % 3], t.vertex[j]);
      Vector uv = Cross(u, v);
      double angle = std::atan2(std::sqrt(Dot(uv, uv)), Dot(u, v));
      normals[a] = Add(normals[a], Mul(t.normal, angle));
    }
  }
  for (const auto &entry : edges)
    if (entry.second.count != 2 || entry.second.orientation != 0)
      return fail("Mesh must have closed, consistently oriented manifold edges");
  if (!(volume > 0) || !std::isfinite(volume))
    return fail("Mesh must have positive volume and outward winding");
  std::vector<grassland::AABB> bounds(indices.size());
  std::vector<int> ids(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    auto &t = mesh->triangles_[i];
    for (int j = 0; j < 3; ++j) {
      int a = indices[i][j], b = indices[i][(j + 1) % 3];
      t.vertex_normal[j] = normals[a];
      t.edge_normal[j] = edges.at(std::minmax(a, b)).normal;
    }
    for (int k = 0; k < 3; ++k) {
      auto coord = [&](Vector v) { return k == 0 ? v.x : k == 1 ? v.y : v.z; };
      // BVHHost uses float bounds. Expand after conversion to avoid pruning a
      // triangle when a bound rounds inward.
      double lo = std::min({coord(t.vertex[0]), coord(t.vertex[1]), coord(t.vertex[2])});
      double hi = std::max({coord(t.vertex[0]), coord(t.vertex[1]), coord(t.vertex[2])});
      bounds[i].lower_bound[k] = std::nextafter(float(lo), -std::numeric_limits<float>::infinity());
      bounds[i].upper_bound[k] = std::nextafter(float(hi), std::numeric_limits<float>::infinity());
    }
    ids[i] = int(i);
  }
  grassland::BVHHost bvh(bounds.data(), ids.data(), int(ids.size()));
  for (const auto &node : bvh.Nodes()) {
    auto a = node.aabb.lower_bound, b = node.aabb.upper_bound;
    mesh->nodes_.push_back(
        {{a[0], a[1], a[2]}, {b[0], b[1], b[2]}, node.instance_index, node.lch, node.next_node_on_failure});
  }
  return mesh;
}

std::unique_ptr<ModelGeometry> ModelGeometry::Load(const Model &model, std::string &error) {
  error.clear();
  auto result = std::make_unique<ModelGeometry>();
  std::unordered_map<std::string, std::shared_ptr<const MeshGeometry>> cache;
  for (const auto &part : model.Parts()) {
    auto found = cache.find(part.mesh);
    if (found == cache.end()) {
      auto mesh = MeshGeometry::LoadObj(part.mesh, error);
      if (!mesh)
        return {};
      found = cache.emplace(part.mesh, std::move(mesh)).first;
    }
    result->meshes_.push_back(found->second);
  }
  return result;
}

bool ModelGeometry::Distance(Vector point,
                             const std::vector<PartPose> &poses,
                             double &distance,
                             std::string &error) const {
  error.clear();
  if (poses.size() != meshes_.size() || poses.empty() || !std::isfinite(point.x) || !std::isfinite(point.y) ||
      !std::isfinite(point.z)) {
    error = "Invalid geometry query or pose count";
    return false;
  }
  double result = 1e300;
  for (size_t i = 0; i < poses.size(); ++i) {
    const auto &transform = poses[i].local_to_body;
    Vector local = transform.InverseRotate(Sub(point, transform.translation));
    if (!std::isfinite(local.x) || !std::isfinite(local.y) || !std::isfinite(local.z)) {
      error = "Nonfinite query transform";
      return false;
    }
    result = std::min(result, meshes_[i]->View().Distance(local));
  }
  distance = result;
  return true;
}
}  // namespace wind3d::aero
