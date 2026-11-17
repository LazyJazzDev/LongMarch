#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

#include "simulation.h"

int main(int argc, char **argv) {
  using namespace wind3d;
  if (argc != 2)
    return 1;
  auto mesh_path = std::filesystem::u8path(argv[1]);
  auto model_path = mesh_path;
  model_path += ".json";
  {
    std::ofstream file(mesh_path);
    file << R"(v -1 -0.5 -0.5
v 1 -0.5 -0.5
v 1 0.5 -0.5
v -1 0.5 -0.5
v -1 -0.5 0.5
v 1 -0.5 0.5
v 1 0.5 0.5
v -1 0.5 0.5
f 4 3 2 1
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
)";
    if (!file)
      return 1;
  }
  {
    std::ofstream file(model_path);
    file
        << "{\"version\":1,\"units\":\"metres\",\"reference_point\":[0.25,0,0],"
           "\"controls\":[{\"name\":\"deflection\",\"min\":-1,\"max\":1,\"default\":0}],"
           "\"parts\":[{\"name\":\"box\",\"mesh\":\""
        << mesh_path.filename().u8string()
        << "\","
           "\"hinge\":{\"pivot\":[0,0,0],\"axis\":[0,0,1],\"offset_deg\":0,\"min_deg\":-45,\"max_deg\":45,\"weights\":{\"deflection\":45}}},"
           "{\"name\":\"fixed\",\"mesh\":\""
        << mesh_path.filename().u8string() << "\",\"translation\":[1.8,0,0]}]}";
    if (!file)
      return 1;
  }
  std::string error;
  SolverOptions options;
  options.model = aero::Model::Load(model_path.u8string(), error);
  options.leapfrog_flow_maps = true;
  options.viscosity = options.wall_viscosity = 1.5e-5f;
  options.wall_model = options.cut_cells = true;
  options.wall_momentum = false;
  options.cfl = .25f;
  options.pressure_cycles = 128;
  options.export_field = false;
  options.retain_snapshots = true;
  options.stop_after = 4;
  options.max_dt = .001f;
  if (!options.model) {
    std::cerr << error;
    return 2;
  }
  auto simulation = Simulation::Create(0, 64, options);
  if (!simulation)
    return 3;
  auto wait = [&](uint64_t step) -> std::shared_ptr<const Snapshot> {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
      if (!simulation->Error().empty()) {
        std::cerr << simulation->Error();
        return {};
      }
      auto s = simulation->TakeNext();
      if (s && s->step == step)
        return s;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return {};
  };
  auto first = wait(4);
  if (!first || !first->generic_geometry || !first->pressure_converged || !first->aircraft ||
      first->aircraft->size() != 72 || !std::isfinite(first->body_loads.force.x) || std::abs(first->force.x) < .01)
    return 4;
  aero::Transform pose;
  if (simulation->ConfigureModel({2}, pose, 12, false, error) || error.empty())
    return 5;
  pose.rotation[0] = -1;
  if (simulation->ConfigureModel({0}, pose, 12, false, error) || error.empty())
    return 6;
  pose = {};
  pose.translation.x = 1000;
  if (simulation->ConfigureModel({0}, pose, 12, false, error) || error.empty())
    return 7;
  pose = {};
  if (!simulation->ConfigureModel({1}, pose, 12, false, error))
    return 8;
  auto second = wait(5);
  if (!second || !second->pressure_converged || second->model_state != std::vector<double>{1} ||
      second->time <= first->time || second->aircraft == first->aircraft)
    return 9;
  auto a = first->aircraft->front().position, b = second->aircraft->front().position;
  if (std::abs(b.x - (a.x - a.y) * std::sqrt(.5)) > 1e-5 || std::abs(b.y - (a.x + a.y) * std::sqrt(.5)) > 1e-5)
    return 10;
  pose.rotation = {0, -1, 0, 1, 0, 0, 0, 0, 1};
  pose.translation = {.5, .25, 0};
  if (!simulation->ConfigureModel({0}, pose, 12, false, error))
    return 11;
  auto third = wait(6);
  if (!third || !third->pressure_converged || third->model_state != std::vector<double>{0})
    return 12;
  auto near = [](double a, double b) { return std::abs(a - b) < 1e-4 * std::max(1.0, std::abs(b)); };
  if (!near(third->body_loads.force.x, third->force.y) || !near(third->body_loads.force.y, -third->force.x) ||
      !near(third->body_loads.moment.z, third->moment.z - .5 * third->force.y + .5 * third->force.x))
    return 13;
  simulation.reset();
  std::error_code ec;
  std::filesystem::remove(mesh_path, ec);
  std::filesystem::remove(model_path, ec);
  std::cout
      << "PASS OBJ fluid boundaries, control warm starts, render mesh poses, body loads, reference torque and rejected states\n";
}
