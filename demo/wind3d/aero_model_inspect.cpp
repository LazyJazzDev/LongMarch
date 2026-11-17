#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "aero_mesh.h"

int main(int argc, char **argv) {
  bool load_meshes = argc > 1 && std::string(argv[1]) == "--geometry";
  int model_arg = load_meshes ? 2 : 1;
  if (argc <= model_arg) {
    std::cerr << "wind3d_model_inspect [--geometry] model.json [control values in declared order]\n";
    return 1;
  }
  std::string error;
  auto model = wind3d::aero::Model::Load(argv[model_arg], error);
  if (!model) {
    std::cerr << error << '\n';
    return 1;
  }
  auto state = model->DefaultState();
  if (argc > model_arg + 1) {
    if (size_t(argc - model_arg - 1) != state.size()) {
      std::cerr << "Wrong number of controls\n";
      return 1;
    }
    for (size_t i = 0; i < state.size(); ++i) {
      char *end;
      state[i] = std::strtod(argv[i + model_arg + 1], &end);
      if (end == argv[i + model_arg + 1] || *end) {
        std::cerr << "Invalid control number\n";
        return 1;
      }
    }
  }
  std::vector<wind3d::aero::PartPose> poses;
  if (!model->Evaluate(state, poses, error)) {
    std::cerr << error << '\n';
    return 1;
  }
  std::unique_ptr<wind3d::aero::ModelGeometry> geometry;
  if (load_meshes) {
    geometry = wind3d::aero::ModelGeometry::Load(*model, error);
    if (!geometry) {
      std::cerr << error << '\n';
      return 1;
    }
  }
  std::cout << std::setprecision(12);
  for (size_t i = 0; i < state.size(); ++i)
    std::cout << "control " << model->Controls()[i].name << '=' << state[i] << '\n';
  for (size_t i = 0; i < poses.size(); ++i) {
    const auto &pose = poses[i];
    std::cout << "part " << model->Parts()[i].name << " mesh=" << model->Parts()[i].mesh
              << " angle_deg=" << pose.angle_degrees << " limited=" << pose.limited << '\n';
    if (geometry)
      std::cout << "triangles=" << geometry->Meshes()[i]->Triangles().size()
                << " bvh_nodes=" << geometry->Meshes()[i]->Nodes().size() << '\n';
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col)
        std::cout << pose.local_to_body.rotation[3 * row + col] << ' ';
      auto t = pose.local_to_body.translation;
      std::cout << (row == 0 ? t.x : row == 1 ? t.y : t.z) << '\n';
    }
  }
}
