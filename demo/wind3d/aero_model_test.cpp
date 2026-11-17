#include "aero_model.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

using namespace wind3d::aero;

namespace {
bool Near(Vector a, Vector b) {
  return std::hypot(a.x - b.x, a.y - b.y, a.z - b.z) < 1e-10;
}

const char *description = R"({
 "version":1,"units":"metres","reference_point":[1,0,0],
 "controls":[{"name":"pitch","min":-1,"max":1,"default":0},
             {"name":"roll","min":-1,"max":1,"default":0}],
 "parts":[
  {"name":"left","mesh":"left.obj","parent":"body","translation":[0,0,1],
   "hinge":{"pivot":[1,0,0],"axis":[0,0,1],"offset_deg":0,"min_deg":-45,"max_deg":45,
            "weights":{"pitch":30,"roll":30}}},
  {"name":"right","mesh":"right.obj","parent":"body","translation":[0,0,-1],
   "hinge":{"pivot":[1,0,0],"axis":[0,0,1],"offset_deg":0,"min_deg":-45,"max_deg":45,
            "weights":{"pitch":30,"roll":-30}}},
  {"name":"body","mesh":"body.obj","translation":[2,3,0],
   "rotation_xyzw":[0,0,0.7071067811865476,0.7071067811865476]}
 ]})";
}  // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  std::filesystem::path path = std::filesystem::u8path(argv[1]);
  auto load = [&](const std::string &json, std::string &error) {
    {
      std::ofstream file(path, std::ios::binary);
      file << json;
      if (!file)
        return std::unique_ptr<Model>{};
    }
    return Model::Load(path.u8string(), error);
  };
  std::string error;
  auto model = load(description, error);
  if (!model) {
    std::cerr << error;
    return 2;
  }
  std::vector<PartPose> poses;
  if (!model->Evaluate({.5, .5}, poses, error) || poses.size() != 3 || std::abs(poses[0].angle_degrees - 30) > 1e-10 ||
      poses[1].angle_degrees != 0 || !Near(poses[0].local_to_body.Point({1, 0, 0}), {2, 4, 1}) ||
      !Near(poses[0].local_to_body.Point({2, 0, 0}), {1.5, 4 + std::sqrt(.75), 1}))
    return 3;
  if (!model->Evaluate({0, 1}, poses, error) || poses[0].angle_degrees != 30 || poses[1].angle_degrees != -30)
    return 4;
  if (!model->Evaluate({1, 0}, poses, error) || poses[0].angle_degrees != 30 || poses[1].angle_degrees != 30)
    return 5;
  if (!model->Evaluate({1, 1}, poses, error) || poses[0].angle_degrees != 45 || !poses[0].limited || poses[1].limited)
    return 6;
  const auto old = poses[0].local_to_body.Point({2, 0, 0});
  if (model->Evaluate({2, 0}, poses, error) || error.empty() || !Near(old, poses[0].local_to_body.Point({2, 0, 0})) ||
      model->Evaluate({0}, poses, error) ||
      model->Evaluate({0, std::numeric_limits<double>::quiet_NaN()}, poses, error))
    return 7;
  auto defaults = model->DefaultState();
  if (defaults != std::vector<double>({0, 0}) || !Near(model->ReferencePoint(), {1, 0, 0}) ||
      std::filesystem::u8path(model->Parts()[0].mesh) != path.parent_path() / "left.obj")
    return 8;
  Transform world;
  world.rotation = {0, -1, 0, 1, 0, 0, 0, 0, 1};
  world.translation = {2, 3, 4};
  auto loads = BodyLoads({{-2, 0, 0}, {0, -8, 8}}, world, {1, 0, 0});
  if (!Near(loads.force, {0, 2, 0}) || !Near(loads.moment, {}))
    return 9;
  loads = BodyLoads({{-2, 0, 0}, {0, -8, 8}}, world, {});
  if (!Near(loads.moment, {0, 0, 2}))
    return 10;
  for (auto change : {std::pair<std::string, std::string>{"\"roll\":30", "\"unknown\":30"},
                      {"\"name\":\"body\"", "\"name\":\"body\",\"parent\":\"left\""},
                      {"\"name\":\"roll\"", "\"name\":\"pitch\""},
                      {"\"axis\":[0,0,1]", "\"axis\":[0,0,0]"},
                      {"\"min\":-1", "\"min\":-1,\"min\":-1"},
                      {"\"units\":\"metres\"", "\"units\":\"feet\""},
                      {"\"version\":1", "\"version\":1,\"typo\":0"}}) {
    std::string invalid = description;
    auto at = invalid.find(change.first);
    if (at == std::string::npos)
      return 11;
    invalid.replace(at, change.first.size(), change.second);
    if (load(invalid, error) || error.empty())
      return 12;
  }
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
  std::cout
      << "PASS mixed controls, limits, hierarchy, pivot invariance, load reference frames and invalid descriptions\n";
}
