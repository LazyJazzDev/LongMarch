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
    << ",\"elevator\":" << c.elevator << ",\"rudder\":" << c.rudder << '}';
}

bool Same(const wind3d::Controls &a, const wind3d::Controls &b) {
  return a.speed == b.speed && a.angle == b.angle && a.flap == b.flap && a.aileron == b.aileron &&
         a.elevator == b.elevator && a.rudder == b.rudder;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "wind3d_web DIRECTORY GRID GPUS\n";
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
  bool paused = false;
  wind3d::Controls requested;
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
      bool valid = (pause == 0 || pause == 1) && controls.speed >= 1 && controls.speed <= 30 && controls.angle >= -20 &&
                   controls.angle <= 25 && controls.flap >= 0 && controls.flap <= 40 &&
                   std::abs(controls.aileron) <= 25 && std::abs(controls.elevator) <= 25 &&
                   std::abs(controls.rudder) <= 25;
      if (valid && sequence != request) {
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
    if (state && now >= next) {
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
        auto lines = wind3d::visualization::FlowLines(*state, true, false);
        bool acknowledged =
            Same(state->controls, requested) && state->generation == expected_generation && actual_paused == paused;
        std::ostringstream json;
        json << std::setprecision(9) << "{\"version\":1,\"frame\":" << ++frame << ",\"step\":" << state->step
             << ",\"generation\":" << state->generation << ",\"mesh\":\"" << mesh_id << "\",\"time\":" << state->time
             << ",\"dt\":" << state->dt << ",\"step_seconds\":" << state->step_seconds
             << ",\"pressure_residual\":" << state->pressure_residual
             << ",\"pressure_converged\":" << (state->pressure_converged ? "true" : "false")
             << ",\"divergence\":" << state->divergence_after << ",\"grid\":" << state->solver_nx
             << ",\"gpus\":" << options.devices << ",\"paused\":" << (actual_paused ? "true" : "false")
             << ",\"ack\":" << (acknowledged ? request : 0) << ",\"line_vertices\":" << lines.size()
             << ",\"mesh_vertices\":" << mesh->size() << ",\"controls\":";
        Controls(json, state->controls);
        json << ",\"solver_model\":\"optimized-13ea18de\",\"force\":";
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
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
  }
  Write(directory / "error.txt", "GPU visualization worker stopped");
}
