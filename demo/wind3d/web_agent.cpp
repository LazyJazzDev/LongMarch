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

#include "aircraft_reference.h"
#include "visualization.h"
#include "web_protocol.h"
#include "web_session.h"

namespace {
std::atomic<bool> running{true};

void Stop(int) {
  running = false;
}

using namespace wind3d::web;

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
  options.export_vorticity = true;
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
  Command initial;
  const bool initial_command = ReadCommand(directory / "controls.txt", initial);
  if (initial_command)
    options.initial_controls = initial.controls;
  auto simulation = wind3d::Simulation::Create(0, std::atoi(argv[2]), options);
  if (!simulation) {
    Write(directory / "error.txt", "CUDA solver initialization failed");
    return 2;
  }
  std::filesystem::remove(directory / "error.txt", error);
  wind3d::web::Session session(directory.string());
  wind3d::Controls requested = options.initial_controls;
  wind3d::ViewSettings view;
  if (initial_command && initial.has_view)
    view = initial.view;
  view.web_directory = directory.string();
  view.streamline_density = initial_command ? initial.density : 2;
  bool paused = initial_command && initial.paused;
  simulation->Configure(requested, paused);
  std::shared_ptr<const wind3d::Snapshot> rendered;
  std::vector<wind3d::Vertex> flow;
  int previous_density = 0;
  bool previous_flow = false;
  while (running) {
    session.Poll(*simulation, requested, paused, view);
    auto failure = simulation->Error();
    if (!failure.empty()) {
      Write(directory / "error.txt", failure);
      return 2;
    }
    auto state = simulation->Latest();
    auto now = std::chrono::steady_clock::now();
    bool actual_paused = simulation->Paused();
    if (state) {
      if (!std::isfinite(state->max_speed) || state->max_speed > 600 || state->dt < 1e-7f ||
          !std::isfinite(wind3d::Length(state->force) + wind3d::Length(state->moment))) {
        simulation->Configure(state->controls, true);
        simulation->Checkpoint((directory / "anomaly.bin").string());
        Write(directory / "error.txt", "Stability guard stopped the run");
        return 2;
      }
      if (state != rendered || previous_density != view.streamline_density || previous_flow != view.flow) {
        flow = view.flow ? wind3d::visualization::DomainFlowLines(*state, view.streamline_density)
                         : std::vector<wind3d::Vertex>{};
        rendered = state;
        previous_density = view.streamline_density;
        previous_flow = view.flow;
      }
      if (!session.Publish(*state, wind3d::HistoryAtSnapshot(simulation->RecentLoads(), *state), view, actual_paused,
                           flow, requested, paused)) {
        Write(directory / "error.txt", "Cannot publish web presentation");
        return 2;
      }
      bool finished = options.stop_time > 0 && state->time >= options.stop_time && actual_paused;
      if (argc >= 6 && (now >= next_checkpoint || finished)) {
        auto path = directory / ("checkpoint-" + session.ID() + "-" + std::to_string(++checkpoint_index) + ".bin");
        if (!simulation->Checkpoint(path.string()))
          return 2;
        next_checkpoint = now + std::chrono::seconds(10);
      }
      if (finished) {
        double wall = std::chrono::duration<double>(now - started).count();
        std::ostringstream result;
        result << "{\"wall_seconds\":" << wall << ",\"physical_seconds\":" << state->time
               << ",\"steps\":" << state->step << "}";
        return Write(directory / "completed.json", result.str()) ? 0 : 2;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
  }
  Write(directory / "error.txt", "GPU visualization worker stopped");
}
