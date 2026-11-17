#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <unordered_set>

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "simulation.h"

namespace {
struct Case {
  std::string name;
  double speed{12};
  double quaternion[4]{0, 0, 0, 1};
  wind3d::aero::Transform pose;
  std::vector<double> state;
};

bool Keys(const rapidjson::Value &v, std::initializer_list<const char *> allowed) {
  if (!v.IsObject())
    return false;
  std::unordered_set<std::string> seen;
  for (auto i = v.MemberBegin(); i != v.MemberEnd(); ++i) {
    std::string key(i->name.GetString(), i->name.GetStringLength());
    if (!seen.insert(key).second ||
        std::none_of(allowed.begin(), allowed.end(), [&](const char *s) { return key == s; }))
      return false;
  }
  return true;
}

bool Number(const rapidjson::Value &v, double &out) {
  if (!v.IsNumber() || !std::isfinite(v.GetDouble()))
    return false;
  out = v.GetDouble();
  return true;
}

std::string Csv(const std::string &value) {
  std::string result = "\"";
  for (char c : value) {
    if (c == '\"')
      result += '\"';
    result += c;
  }
  return result + '\"';
}

bool ReadCases(const std::string &path,
               const wind3d::aero::Model &model,
               std::vector<Case> &cases,
               std::string &error) {
  std::ifstream file(std::filesystem::u8path(path));
  if (!file) {
    error = "Cannot open cases file";
    return false;
  }
  rapidjson::IStreamWrapper stream(file);
  rapidjson::Document root;
  root.ParseStream<rapidjson::kParseValidateEncodingFlag>(stream);
  if (root.HasParseError() || !Keys(root, {"version", "cases"}) || !root.HasMember("version") || root["version"] != 1 ||
      !root.HasMember("cases") || !root["cases"].IsArray() || root["cases"].Empty()) {
    error = "Expected version 1 and a nonempty cases array";
    return false;
  }
  std::unordered_set<std::string> names;
  for (const auto &v : root["cases"].GetArray()) {
    if (!Keys(v, {"name", "speed", "controls", "rotation_xyzw", "translation"}) || !v.HasMember("name") ||
        !v["name"].IsString()) {
      error = "Invalid case fields or missing name";
      return false;
    }
    Case c;
    c.name.assign(v["name"].GetString(), v["name"].GetStringLength());
    c.state = model.DefaultState();
    if (!names.insert(c.name).second || c.name.empty() || c.name.find_first_of("\r\n\0", 0, 3) != std::string::npos) {
      error = "Invalid case name";
      return false;
    }
    if (v.HasMember("speed") && (!Number(v["speed"], c.speed) || !(c.speed > 0) || c.speed > 1000)) {
      error = "Invalid inlet speed";
      return false;
    }
    if (v.HasMember("controls")) {
      const auto &a = v["controls"];
      if (!a.IsArray() || a.Size() != c.state.size()) {
        error = "Wrong control count";
        return false;
      }
      for (size_t i = 0; i < c.state.size(); ++i)
        if (!Number(a[rapidjson::SizeType(i)], c.state[i])) {
          error = "Invalid control value";
          return false;
        }
    }
    if (v.HasMember("rotation_xyzw")) {
      const auto &q = v["rotation_xyzw"];
      if (!q.IsArray() || q.Size() != 4) {
        error = "Expected rotation_xyzw quaternion";
        return false;
      }
      for (int i = 0; i < 4; ++i)
        if (!Number(q[i], c.quaternion[i])) {
          error = "Invalid quaternion";
          return false;
        }
      double x = c.quaternion[0], y = c.quaternion[1], z = c.quaternion[2], w = c.quaternion[3];
      if (std::abs(x * x + y * y + z * z + w * w - 1) > 1e-8) {
        error = "Quaternion must be unit length";
        return false;
      }
      c.pose.rotation = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
                         2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                         2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)};
    }
    if (v.HasMember("translation")) {
      const auto &t = v["translation"];
      if (!t.IsArray() || t.Size() != 3 || !Number(t[0], c.pose.translation.x) || !Number(t[1], c.pose.translation.y) ||
          !Number(t[2], c.pose.translation.z)) {
        error = "Invalid body translation";
        return false;
      }
    }
    std::vector<wind3d::aero::PartPose> poses;
    if (!model.Evaluate(c.state, poses, error))
      return false;
    cases.push_back(std::move(c));
  }
  return true;
}

bool Matches(const wind3d::Snapshot &s, const Case &c) {
  return s.generic_geometry && s.model_state == c.state && s.controls.speed == float(c.speed) &&
         s.body_to_world.rotation == c.pose.rotation && s.body_to_world.translation.x == c.pose.translation.x &&
         s.body_to_world.translation.y == c.pose.translation.y && s.body_to_world.translation.z == c.pose.translation.z;
}
}  // namespace

int main(int argc, char **argv) {
  using namespace wind3d;
  if (argc < 4) {
    std::cerr
        << "wind3d_model_sample model.json cases.json output.csv [--grid N] [--settle seconds] [--average seconds] [--maccormack]\n";
    return 1;
  }
  int grid = 128;
  double settle = .5, average = .5;
  SolverOptions options;
  options.leapfrog_flow_maps = true;
  options.viscosity = options.wall_viscosity = 1.5e-5f;
  options.wall_model = options.cut_cells = true;
  options.wall_momentum = false;
  options.sponge = 2;
  options.cfl = .25f;
  options.pressure_cycles = 128;
  options.display = options.export_field = false;
  options.retain_snapshots = true;
  for (int i = 4; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--maccormack") {
      options.leapfrog_flow_maps = false;
      continue;
    }
    if (++i == argc) {
      std::cerr << "Missing option value\n";
      return 1;
    }
    char *end;
    double value = std::strtod(argv[i], &end);
    if (end == argv[i] || *end || !std::isfinite(value)) {
      std::cerr << "Invalid number\n";
      return 1;
    }
    if (key == "--grid" && value >= 64 && value <= max_grid && std::floor(value) == value && int(value) % 16 == 0)
      grid = int(value);
    else if (key == "--settle" && value >= 0)
      settle = value;
    else if (key == "--average" && value > 0)
      average = value;
    else {
      std::cerr << "Invalid option " << key << '\n';
      return 1;
    }
  }
  if (!std::isfinite(settle + average)) {
    std::cerr << "Invalid sampling duration\n";
    return 1;
  }
  std::string error;
  options.model = aero::Model::Load(argv[1], error);
  if (!options.model) {
    std::cerr << error << '\n';
    return 1;
  }
  std::vector<Case> cases;
  if (!ReadCases(argv[2], *options.model, cases, error)) {
    std::cerr << error << '\n';
    return 1;
  }
  // Avoid silently replacing an existing dataset or an input file.
  auto output_path = std::filesystem::u8path(argv[3]);
  std::error_code ec;
  if (std::filesystem::exists(output_path, ec) || ec) {
    std::cerr << "Output path must not exist\n";
    return 1;
  }
  auto partial_path = output_path;
  partial_path += ".partial";
  if (std::filesystem::exists(partial_path, ec) || ec) {
    std::cerr << "Partial output already exists\n";
    return 1;
  }
  std::ofstream output(partial_path, std::ios::binary);
  if (!output) {
    std::cerr << "Cannot create dataset\n";
    return 1;
  }
  output << std::setprecision(17)
         << "case,grid,lfm,speed,vx_body,vy_body,vz_body,qx,qy,qz,qw,tx,ty,tz,reference_x,reference_y,reference_z";
  for (const auto &control : options.model->Controls())
    output << ',' << Csv("control:" + control.name);
  output
      << ",window_start_seconds,window_end_seconds,fx_body,fy_body,fz_body,mx_body,my_body,mz_body,force_rms,moment_rms,pressure_residual_max,wall_seconds\n";
  options.initial_controls.speed = float(cases[0].speed);
  options.initial_model_state = cases[0].state;
  options.initial_body_pose = cases[0].pose;
  auto simulation = Simulation::Create(0, grid, options);
  if (!simulation) {
    std::cerr << "Solver initialization failed\n";
    return 1;
  }
  for (const auto &c : cases) {
    uint64_t after = simulation->Latest() ? simulation->Latest()->step : 0;
    if (!simulation->ConfigureModel(c.state, c.pose, float(c.speed), false, error)) {
      std::cerr << error << '\n';
      return 1;
    }
    auto started = std::chrono::steady_clock::now();
    double start = -1, total = 0, sum[6]{}, squares[2]{}, residual = 0, previous_time = 0;
    uint64_t previous_step = 0;
    while (total < average * (1 - 1e-10)) {
      if (!(error = simulation->Error()).empty()) {
        std::cerr << error << '\n';
        return 1;
      }
      auto sample = simulation->TakeNext();
      if (!sample) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      if (sample->step <= after || !Matches(*sample, c))
        continue;
      if (!sample->pressure_converged) {
        std::cerr << "Unconverged projection in " << c.name << '\n';
        return 2;
      }
      if (previous_step && sample->step != previous_step + 1) {
        std::cerr << "Missing solver snapshot\n";
        return 2;
      }
      // Accumulate per-case physical time in double precision; a long batch
      // must not inherit rounding from the solver display clock.
      double left = previous_time, right = previous_time + sample->dt;
      previous_step = sample->step;
      previous_time = right;
      if (start < 0) {
        start = left + settle;
        if (!(start + average > start) || !std::isfinite(start + average)) {
          std::cerr << "Sampling window cannot be represented\n";
          return 1;
        }
      }
      double weight = std::max(0.0, std::min(right, start + average) - std::max(left, start));
      if (weight <= 0)
        continue;
      const auto &l = sample->body_loads;
      double values[] = {l.force.x, l.force.y, l.force.z, l.moment.x, l.moment.y, l.moment.z};
      for (int k = 0; k < 6; ++k) {
        if (!std::isfinite(values[k])) {
          std::cerr << "Nonfinite load\n";
          return 2;
        }
        sum[k] += weight * values[k];
        squares[k / 3] += weight * values[k] * values[k];
      }
      total += weight;
      residual = std::max(residual, double(sample->pressure_residual));
      if (right >= start + average)
        break;
    }
    if (!simulation->ConfigureModel(c.state, c.pose, float(c.speed), true, error)) {
      std::cerr << error;
      return 1;
    }
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    auto reference = options.model->ReferencePoint();
    output << Csv(c.name) << ',' << grid << ',' << options.leapfrog_flow_maps << ',' << float(c.speed);
    auto flow = c.pose.InverseRotate({float(c.speed), 0, 0});
    output << ',' << flow.x << ',' << flow.y << ',' << flow.z;
    for (double q : c.quaternion)
      output << ',' << q;
    output << ',' << c.pose.translation.x << ',' << c.pose.translation.y << ',' << c.pose.translation.z << ','
           << reference.x << ',' << reference.y << ',' << reference.z;
    for (double u : c.state)
      output << ',' << u;
    output << ',' << start << ',' << start + average;
    for (double v : sum)
      output << ',' << v / total;
    for (int j = 0; j < 2; ++j) {
      double variance = squares[j] / total;
      for (int k = 0; k < 3; ++k)
        variance -= std::pow(sum[3 * j + k] / total, 2);
      output << ',' << std::sqrt(std::max(0.0, variance));
    }
    output << ',' << residual << ',' << wall << '\n';
    output.flush();
    if (!output) {
      std::cerr << "Dataset write failed\n";
      return 1;
    }
    std::cout << c.name << ": body force=" << sum[0] / total << ',' << sum[1] / total << ',' << sum[2] / total
              << " moment=" << sum[3] / total << ',' << sum[4] / total << ',' << sum[5] / total
              << " wall_seconds=" << wall << std::endl;
  }
  output.close();
  if (!output) {
    std::cerr << "Dataset close failed\n";
    return 1;
  }
  std::filesystem::rename(partial_path, output_path, ec);
  if (ec) {
    std::cerr << "Cannot publish dataset: " << ec.message() << '\n';
    return 1;
  }
}
