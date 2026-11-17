#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

#include "visualization.h"

namespace {
std::atomic<bool> running{true};

void Stop(int) {
  running = false;
}

bool Write(const std::filesystem::path &path, const std::string &bytes) {
  auto temporary = path.string() + ".tmp";
  std::ofstream file(temporary, std::ios::binary);
  file.write(bytes.data(), bytes.size());
  file.close();
  if (!file)
    return false;
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  return !error;
}

void Vector(std::ostream &s, wind3d::Vec3 v) {
  s << '[' << v.x << ',' << v.y << ',' << v.z << ']';
}

void Controls(std::ostream &s, const wind3d::Controls &c) {
  s << "{\"speed\":" << c.speed << ",\"angle\":" << c.angle << ",\"flap\":" << c.flap << ",\"aileron\":" << c.aileron
    << ",\"elevator\":" << c.elevator << ",\"rudder\":" << c.rudder
    << ",\"tail_incidence\":" << c.airframe.tail_incidence << ",\"winglets\":" << int(c.airframe.winglets)
    << ",\"linked_flaps\":" << int(c.wing_controls.linked_flaps) << ",\"outer_flap\":" << c.wing_controls.outer_flap
    << ",\"cruise_camber\":" << c.wing_controls.cruise_camber << ",\"aileron_droop\":" << c.wing_controls.aileron_droop
    << ",\"spoiler\":" << c.wing_controls.spoiler << ",\"spoiler_roll\":" << c.wing_controls.spoiler_roll
    << ",\"linked_spoiler_roll\":" << int(c.wing_controls.linked_spoiler_roll) << ",\"slat\":" << c.wing_controls.slat
    << ",\"linked_slats\":" << int(c.wing_controls.linked_slats) << '}';
}

bool Same(const wind3d::Controls &a, const wind3d::Controls &b) {
  return a.speed == b.speed && wind3d::SameAircraftGeometry(a, b);
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 4 && argc != 6 && argc != 7) {
    std::cerr
        << "wind3d_web DIRECTORY GRID GPUS [aero|baseline|linear-cut|linear-slip|viscous STOP_TIME [CHECKPOINT]]\n";
    return 1;
  }
  std::filesystem::path directory(argv[1]);
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error)
    return 1;
  std::signal(SIGTERM, Stop);
  std::signal(SIGINT, Stop);
  auto options = wind3d::OptimizedSolverOptions();
  options.devices = std::atoi(argv[3]);
  options.record_loads = true;
  std::string profile = argc >= 6 ? argv[4] : "baseline";
  if (profile != "baseline" && profile != "linear-cut" && profile != "linear-slip" && profile != "viscous" &&
      profile != "aero")
    return 1;
  if (argc >= 6) {
    if (options.devices != 1)
      return 1;
    options.stop_time = std::atof(argv[5]);
    if (!(options.stop_time >= 0) || !std::isfinite(options.stop_time))
      return 1;
    options.initial_controls.speed = 30;
    options.initial_controls.angle = 6;
    options.diagnostics_path = (directory / "diagnostics.csv").string();
    if (profile != "baseline" && profile != "aero") {
      options.corrected_transport = false;
      options.cubic = false;
      options.cut_cells = profile == "linear-cut";
      options.cfl = 0.5f;
      options.pressure_cycles = 128;
    }
    if (profile == "viscous") {
      options.wall_model = false;
      options.wall_momentum = true;
      options.viscosity = 0.005f;
    }
  }
  if (profile == "aero") {
    options.leapfrog_flow_maps = true;
    options.viscosity = options.wall_viscosity = 1.5e-5f;
    options.wall_model = options.cut_cells = true;
    options.wall_momentum = false;
    options.cubic = false;
    options.sponge = 2;
    options.cfl = .25f;
    options.pressure_cycles = 128;
  }
  if (argc == 7)
    options.initial_checkpoint = argv[6];
  const auto started = std::chrono::steady_clock::now();
  auto next_checkpoint = started;
  unsigned checkpoint_index = 0;
  auto simulation = wind3d::Simulation::Create(0, std::atoi(argv[2]), options);
  if (!simulation) {
    Write(directory / "error.txt", "CUDA solver initialization failed");
    return 2;
  }
  std::filesystem::remove(directory / "error.txt", error);
  uint64_t request = 0, reset = 0, expected_generation = 0, last_step = 0, last_generation = ~uint64_t(0);
  uint64_t frame = 0;
  // Unique mesh IDs survive restarting the agent in the same mailbox.
  uint64_t mesh_id = std::chrono::system_clock::now().time_since_epoch().count();
  const auto run_id = std::to_string(mesh_id);
  uint64_t published_load_step = 0, published_load_generation = ~uint64_t(0);
  auto next_loads = std::chrono::steady_clock::now();
  bool paused = false;
  int density = 2;
  wind3d::Controls requested = options.initial_controls;
  std::shared_ptr<const std::vector<wind3d::Vertex>> mesh;

  struct Sample {
    double time;
    std::array<double, 3> impulse;
  };

  std::deque<Sample> history;
  wind3d::Controls history_controls;
  uint64_t history_generation = ~uint64_t(0);
  double condition_start = 0;
  auto next = std::chrono::steady_clock::now();
  while (running) {
    std::ifstream input(directory / "controls.txt");
    uint64_t sequence, reset_request;
    int pause;
    wind3d::Controls controls;
    if (input >> sequence >> reset_request >> pause >> controls.speed >> controls.angle >> controls.flap >>
        controls.aileron >> controls.elevator >> controls.rudder) {
      int requested_density = 2;
      input >> requested_density;
      bool extension_valid = true;
      std::string extension;
      if (input >> extension) {
        int winglets = 1, linked_flaps = 1, linked_roll = 0, linked_slats = 0;
        auto &w = controls.wing_controls;
        extension_valid =
            extension == "wing-v1" && bool(input >> controls.airframe.tail_incidence >> winglets >> linked_flaps >>
                                           w.outer_flap >> w.cruise_camber >> w.aileron_droop >> w.spoiler >>
                                           w.spoiler_roll >> linked_roll >> w.slat >> linked_slats);
        auto flag = [](int value) { return value == 0 || value == 1; };
        extension_valid &= flag(winglets) && flag(linked_flaps) && flag(linked_roll) && flag(linked_slats);
        controls.airframe.winglets = winglets != 0;
        w.linked_flaps = linked_flaps != 0;
        w.linked_spoiler_roll = linked_roll != 0;
        w.linked_slats = linked_slats != 0;
      }
      bool valid = extension_valid && wind3d::ValidAirframe(controls.airframe) &&
                   wind3d::ValidWingControls(controls.wing_controls) && requested_density >= 1 &&
                   requested_density <= 4 && (pause == 0 || pause == 1) && controls.speed >= 1 &&
                   controls.speed <= 30 && controls.angle >= -20 && controls.angle <= 25 && controls.flap >= 0 &&
                   controls.flap <= 40 && std::abs(controls.aileron) <= 25 && std::abs(controls.elevator) <= 25 &&
                   std::abs(controls.rudder) <= 25;
      if (valid && sequence != request) {
        density = requested_density;
        requested = controls;
        paused = pause != 0;
        request = sequence;
        simulation->Configure(requested, paused);
        if (reset_request != reset) {
          simulation->Reset();
          ++expected_generation;
          reset = reset_request;
        }
      }
    }
    auto failure = simulation->Error();
    if (!failure.empty()) {
      Write(directory / "error.txt", failure);
      return 2;
    }
    bool actual_paused = simulation->Paused();
    auto state = simulation->Latest();
    auto now = std::chrono::steady_clock::now();
    if (state && argc >= 6 &&
        (!std::isfinite(state->max_speed) || state->max_speed > 600 || state->dt < 1e-7f ||
         !std::isfinite(state->force.x + state->force.y + state->force.z + state->moment.x + state->moment.y +
                        state->moment.z))) {
      simulation->Configure(state->controls, true);
      simulation->Checkpoint((directory / "anomaly.bin").string());
      Write(directory / "error.txt", "Stability guard stopped the run; see raw diagnostics and checkpoints");
      return 2;
    }
    bool finished = state && options.stop_time > 0 && state->time >= options.stop_time && actual_paused;
    if (now >= next_loads || finished) {
      const auto loads = simulation->RecentLoads();
      if (!loads.samples.empty() &&
          (loads.samples.back().step != published_load_step || loads.generation != published_load_generation)) {
        std::ostringstream header;
        header << std::setprecision(9) << "{\"version\":1,\"run\":\"" << run_id
               << "\",\"generation\":" << loads.generation << ",\"count\":" << loads.samples.size()
               << ",\"stride\":44,\"retention_seconds\":2,\"capacity_limited\":"
               << (loads.capacity_limited ? "true" : "false") << ",\"controls_events\":[";
        bool first = true;
        wind3d::Controls previous;
        for (const auto &sample : loads.samples) {
          if (first || !Same(sample.controls, previous)) {
            if (!first)
              header << ',';
            header << "{\"step\":" << sample.step << ",\"time\":" << sample.time << ",\"controls\":";
            Controls(header, sample.controls);
            header << '}';
            previous = sample.controls;
            first = false;
          }
        }
        header << "]}";
        auto json = header.str();
        uint32_t length = uint32_t(json.size());
        std::string packet(reinterpret_cast<const char *>(&length), sizeof(length));
        packet += json;
        for (const auto &sample : loads.samples) {
          packet.append(reinterpret_cast<const char *>(&sample.step), sizeof(sample.step));
          packet.append(reinterpret_cast<const char *>(&sample.time), sizeof(sample.time));
          packet.append(reinterpret_cast<const char *>(&sample.dt), sizeof(sample.dt));
          packet.append(reinterpret_cast<const char *>(&sample.force), sizeof(sample.force));
          packet.append(reinterpret_cast<const char *>(&sample.moment), sizeof(sample.moment));
        }
        if (!Write(directory / "loads.bin", packet))
          return 2;
        published_load_step = loads.samples.back().step;
        published_load_generation = loads.generation;
      }
      next_loads = now + std::chrono::seconds(1);
    }
    if (state && (now >= next || finished)) {
      bool changed = state->step != last_step || state->generation != last_generation;
      // Refresh paused metadata as well, so pause acknowledgements and liveness remain visible.
      if (changed || paused) {
        if (mesh != state->aircraft) {
          mesh = state->aircraft;
          ++mesh_id;
          std::string bytes(reinterpret_cast<const char *>(mesh->data()), mesh->size() * sizeof(wind3d::Vertex));
          if (!Write(directory / ("mesh-" + std::to_string(mesh_id) + ".bin"), bytes))
            return 2;
          // Keep recent immutable meshes for clients that are finishing a frame fetch.
          std::vector<std::filesystem::path> files;
          for (const auto &entry : std::filesystem::directory_iterator(directory))
            if (entry.path().filename().string().rfind("mesh-", 0) == 0 && entry.path().extension() == ".bin")
              files.push_back(entry.path());
          std::sort(files.begin(), files.end());
          while (files.size() > 32) {
            std::filesystem::remove(files.front(), error);
            files.erase(files.begin());
          }
        }
        if (history_generation != state->generation || !Same(history_controls, state->controls)) {
          history.clear();
          history_generation = state->generation;
          history_controls = state->controls;
          condition_start = state->time;
        }
        if (history.empty() || history.back().time < state->time)
          history.push_back({state->time, {state->impulse[0], state->impulse[1], state->impulse[2]}});
        while (history.size() > 2 && history[1].time <= state->time - 0.15)
          history.pop_front();
        double average_span = state->time - history.front().time;
        bool average_ready = state->time - condition_start >= 0.55 && average_span >= 0.15;
        auto lines = wind3d::visualization::FlowLines(*state, true, false, density);
        bool acknowledged =
            Same(state->controls, requested) && state->generation == expected_generation && actual_paused == paused;
        std::ostringstream json;
        json << std::setprecision(9) << "{\"version\":1,\"frame\":" << ++frame << ",\"step\":" << state->step
             << ",\"generation\":" << state->generation << ",\"mesh\":\"" << mesh_id << "\",\"time\":" << state->time
             << ",\"dt\":" << state->dt << ",\"step_seconds\":" << state->step_seconds
             << ",\"max_speed\":" << state->max_speed << ",\"pressure_cycles\":" << state->pressure_cycles
             << ",\"pressure_residual\":" << state->pressure_residual
             << ",\"pressure_converged\":" << (state->pressure_converged ? "true" : "false")
             << ",\"divergence\":" << state->divergence_after << ",\"grid\":" << state->solver_nx
             << ",\"gpus\":" << options.devices << ",\"paused\":" << (actual_paused ? "true" : "false")
             << ",\"ack\":" << (acknowledged ? request : 0) << ",\"line_vertices\":" << lines.size()
             << ",\"mesh_vertices\":" << mesh->size() << ",\"controls\":";
        Controls(json, state->controls);
        json << ",\"streamline_density\":" << density
             << ",\"streamline_seeds\":" << (20 * density + 1) * (3 * density + 1)
             << ",\"domain_min\":[-6,-4,-6],\"domain_max\":[10,4,6]";
        json << ",\"solver_model\":\"uniform-" << profile
             << "\",\"geometry_model\":\"narrowbody-junctions-slats-spoilers\",\"force\":";
        Vector(json, state->force);
        json << ",\"moment\":";
        Vector(json, state->moment);
        json << ",\"average_seconds\":" << average_span << ",\"mean_ld\":";
        double drag = state->impulse[0] - history.front().impulse[0];
        if (average_ready && drag > 1e-9)
          json << (state->impulse[1] - history.front().impulse[1]) / drag;
        else
          json << "null";
        json << '}';
        auto header = json.str();
        while (header.size() % 4)
          header += ' ';
        uint32_t length = uint32_t(header.size());
        std::string packet(reinterpret_cast<const char *>(&length), sizeof(length));
        packet += header;
        // Only line position/color reaches shared storage, never the velocity/pressure volume.
        for (const auto &v : lines) {
          packet.append(reinterpret_cast<const char *>(&v.position), sizeof(v.position));
          packet.append(reinterpret_cast<const char *>(&v.color), sizeof(v.color));
        }
        if (!Write(directory / "frame.bin", packet))
          return 2;
        last_step = state->step;
        last_generation = state->generation;
        if (frame % 40 == 1)
          std::cout << "frame=" << frame << " step=" << state->step << " bytes=" << packet.size() << std::endl;
      }
      next = now + std::chrono::milliseconds(500);
    }
    if (state && argc >= 6 && (now >= next_checkpoint || finished)) {
      auto path = directory / ("checkpoint-" + run_id + "-" + std::to_string(++checkpoint_index) + ".bin");
      auto checkpoint_step = simulation->Checkpoint(path.string());
      if (!checkpoint_step) {
        Write(directory / "error.txt", "Checkpoint failed");
        return 2;
      }
      next_checkpoint = std::chrono::steady_clock::now() + std::chrono::seconds(10);
      std::cout << "checkpoint=" << path << " step=" << checkpoint_step << std::endl;
    }
    if (finished) {
      double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      std::ostringstream result;
      result << std::setprecision(17) << "{\"wall_seconds\":" << wall << ",\"physical_seconds\":" << state->time
             << ",\"steps\":" << state->step << ",\"gpus\":" << options.devices << ",\"profile\":\"" << profile
             << "\"}";
      if (!Write(directory / "completed.json", result.str()))
        return 2;
      return 0;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
  }
  Write(directory / "error.txt", "GPU visualization worker stopped");
}
