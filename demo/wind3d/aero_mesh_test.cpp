#include "aero_mesh.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

using namespace wind3d::aero;
#ifdef WIND3D_MESH_CUDA_TEST
bool CheckCudaMesh(const MeshGeometry &mesh, const std::vector<Vector> &points, const std::vector<double> &expected);
#endif
namespace {
std::string Voxels(const std::set<std::array<int, 3>> &cells) {
  std::ostringstream out;
  int count = 0;
  for (auto cell : cells)
    for (int axis = 0; axis < 3; ++axis)
      for (int sign : {-1, 1}) {
        auto adjacent = cell;
        adjacent[axis] += sign;
        if (cells.count(adjacent))
          continue;
        int u = (axis + 1) % 3, v = (axis + 2) % 3;
        for (auto corner : {std::array<int, 2>{0, 0}, {1, 0}, {1, 1}, {0, 1}}) {
          auto p = cell;
          p[axis] += sign > 0;
          p[u] += corner[0];
          p[v] += corner[1];
          out << "v " << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
        }
        if (sign > 0)
          out << "f " << count + 1 << ' ' << count + 2 << ' ' << count + 3 << ' ' << count + 4 << '\n';
        else
          out << "f " << count + 4 << ' ' << count + 3 << ' ' << count + 2 << ' ' << count + 1 << '\n';
        count += 4;
      }
  return out.str();
}

double BoxDistance(Vector p) {
  double x = std::abs(p.x - .5) - .5, y = std::abs(p.y - .5) - .5, z = std::abs(p.z - .5) - .5;
  return std::hypot(std::max(x, 0.0), std::max(y, 0.0), std::max(z, 0.0)) + std::min(std::max({x, y, z}), 0.0);
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  auto path = std::filesystem::u8path(argv[1]);
  std::string error;
  auto load = [&](const std::string &text) {
    {
      std::ofstream out(path, std::ios::binary);
      out << text;
      if (!out)
        return std::shared_ptr<const MeshGeometry>{};
    }
    return MeshGeometry::LoadObj(path.u8string(), error);
  };
  auto cube_text = Voxels({{0, 0, 0}});
  auto cube = load(cube_text);
  if (!cube) {
    std::cerr << error;
    return 2;
  }
  std::vector<Vector> points;
  std::vector<double> expected;
  for (int x = -8; x <= 16; ++x)
    for (int y = -8; y <= 16; ++y)
      for (int z = -8; z <= 16; ++z) {
        Vector p{x / 8.0, y / 8.0, z / 8.0};
        double want = BoxDistance(p), got = cube->View().Distance(p);
        if (std::abs(want - got) > 1e-10) {
          std::cerr << "Cube distance " << want << ' ' << got;
          return 3;
        }
        points.push_back(p);
        expected.push_back(want);
      }
#ifdef WIND3D_MESH_CUDA_TEST
  if (!CheckCudaMesh(*cube, points, expected))
    return 4;
#endif
  auto concave = load(Voxels({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}));
  if (!concave) {
    std::cerr << error;
    return 5;
  }
  points.clear();
  expected.clear();
  for (int x = -3; x < 15; ++x)
    for (int y = -3; y < 15; ++y)
      for (int z = -3; z < 9; ++z) {
        Vector p{(x + .37) / 6, (y + .41) / 6, (z + .23) / 6};
        bool inside = p.x > 0 && p.y > 0 && p.z > 0 && p.z < 1 && p.x < 2 && p.y < 2 && (p.x < 1 || p.y < 1);
        double got = concave->View().Distance(p);
        MeshHit brute;
        for (size_t i = 0; i < concave->Triangles().size(); ++i)
          TriangleHit(concave->Triangles()[i], p, int(i), brute);
        double want = (inside ? -1 : 1) * std::sqrt(brute.squared_distance);
        if (std::abs(got - want) > 1e-10) {
          std::cerr << "Concave distance " << want << ' ' << got;
          return 6;
        }
        points.push_back(p);
        expected.push_back(want);
      }
#ifdef WIND3D_MESH_CUDA_TEST
  if (!CheckCudaMesh(*concave, points, expected))
    return 7;
#endif
  // Two instances share one immutable mesh. A rotated, translated pose changes
  // spatial queries without touching its BVH.
  cube = load(cube_text);
  auto json_path = path;
  json_path += ".json";
  {
    std::ofstream out(json_path);
    out << "{\"version\":1,\"units\":\"metres\",\"reference_point\":[0,0,0],\"controls\":[],\"parts\":["
        << "{\"name\":\"a\",\"mesh\":\"" << path.filename().u8string() << "\"},"
        << "{\"name\":\"b\",\"mesh\":\"" << path.filename().u8string()
        << "\",\"translation\":[3,0,0],\"rotation_xyzw\":[0,0,0.7071067811865476,0.7071067811865476]}]}";
  }
  auto model = Model::Load(json_path.u8string(), error);
  if (!model)
    return 8;
  auto geometry = ModelGeometry::Load(*model, error);
  std::vector<PartPose> poses;
  if (!geometry || !model->Evaluate({}, poses, error) || geometry->Meshes()[0] != geometry->Meshes()[1])
    return 9;
  double distance;
  if (!geometry->Distance({2.5, .5, .5}, poses, distance, error) || std::abs(distance + .5) > 1e-10)
    return 10;
  if (!geometry->Distance({1.5, .5, .5}, poses, distance, error) || std::abs(distance - .5) > 1e-10)
    return 11;
  auto open = cube_text.substr(0, cube_text.rfind("f "));
  if (load(open) || error.empty())
    return 12;
  if (load(cube_text + "f 1 1 1\n") || error.empty())
    return 13;
  if (MeshGeometry::LoadObj(path.u8string() + ".missing", error) || error.empty())
    return 14;
  std::error_code ec;
  std::filesystem::remove(path, ec);
  std::filesystem::remove(json_path, ec);
  std::cout
      << "PASS analytic cube, concave interior, BVH/brute force, shared instances, transformed queries, invalid meshes\n";
}
