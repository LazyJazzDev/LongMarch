// Headless single-GPU control-surface sweep for the Wind3D aircraft.
//
// One CUDA worker advances the flow continuously. Each case changes the
// controls at a solver-step boundary, starting from the previous case's flow,
// waits for the transient and averages the loads over a window. Averages use
// the solver's time-integrated force and moment, so no step is lost between
// polls of the latest snapshot.
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "aircraft_reference.h"
#include "validation.h"

namespace {
using wind3d::Controls;

struct Case {
  std::string name;
  Controls controls;
};

bool ParseCases(const std::string &path, const Controls &base, std::vector<Case> &cases) {
  std::ifstream file(path);
  if (!file)
    return false;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream row(line);
    Case c{{}, base};
    if (!(row >> c.name >> c.controls.angle >> c.controls.flap >> c.controls.aileron >> c.controls.elevator >>
          c.controls.rudder)) {
      std::cerr << "Invalid case line: " << line << '\n';
      return false;
    }
    cases.push_back(c);
  }
  return !cases.empty();
}

std::vector<Case> Preset(const std::string &name, const Controls &base) {
  std::vector<Case> cases;
  auto add = [&](const std::string &label, float angle, float flap, float aileron, float elevator, float rudder) {
    Controls c = base;
    c.angle = angle;
    c.flap = flap;
    c.aileron = aileron;
    c.elevator = elevator;
    c.rudder = rudder;
    cases.push_back({label, c});
  };
  float a = base.angle;
  if (name == "polar") {
    for (float angle : {-2.f, 0.f, 2.f, 4.f, 6.f, 8.f, 10.f})
      add("alpha" + std::to_string(int(angle)), angle, 0, 0, 0, 0);
  } else if (name == "controls") {
    add("baseline", a, 0, 0, 0, 0);
    for (float flap : {10.f, 20.f, 30.f, 40.f})
      add("flap" + std::to_string(int(flap)), a, flap, 0, 0, 0);
    for (float aileron : {-20.f, -10.f, 10.f, 20.f})
      add("aileron" + std::to_string(int(aileron)), a, 0, aileron, 0, 0);
    for (float elevator : {-20.f, -10.f, 10.f, 20.f})
      add("elevator" + std::to_string(int(elevator)), a, 0, 0, elevator, 0);
    for (float rudder : {-20.f, -10.f, 10.f, 20.f})
      add("rudder" + std::to_string(int(rudder)), a, 0, 0, 0, rudder);
    add("baseline_repeat", a, 0, 0, 0, 0);
  } else if (name == "cruise") {
    add("baseline", a, 0, 0, 0, 0);
  }
  return cases;
}

int LargestGrid(int device) {
  size_t free_bytes = 0, total_bytes = 0;
  if (cudaSetDevice(device) != cudaSuccess || cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
    return 0;
  int best = 0;
  for (int nx = 64; nx <= wind3d::max_grid; nx += 16)
    if (double(wind3d::SolverBytes(nx)) <= 0.97 * double(free_bytes))
      best = nx;
  return best;
}

struct Window {
  double time{}, impulse[3]{}, angular[3]{}, friction{}, drag[wind3d::kComponents]{}, lift[wind3d::kComponents]{};
};

Window Mark(const wind3d::Snapshot &s) {
  Window w;
  w.time = s.time;
  for (int k = 0; k < 3; ++k) {
    w.impulse[k] = s.impulse[k];
    w.angular[k] = s.angular_impulse[k];
  }
  w.friction = s.friction_impulse[0];
  for (int k = 0; k < wind3d::kComponents; ++k) {
    w.drag[k] = s.drag_impulse[k];
    w.lift[k] = s.lift_impulse[k];
  }
  return w;
}

constexpr int quarters = 4;

// Sweep position saved next to a solver checkpoint, so a preempted job can
// resume mid-case. `step` must equal the checkpoint's step.
struct Progress {
  size_t index{};
  double switch_time{-1};
  int next_mark{}, max_cycles{};
  float max_residual{};
  uint64_t first_step{}, last_step{}, step{};
  double base[6]{};
  Window marks[quarters + 1];
};

bool SaveProgress(const std::string &path, const Progress &p) {
  std::string temporary = path + ".tmp";
  {
    std::ofstream file(temporary);
    file.precision(17);
    file << p.index << ' ' << p.switch_time << ' ' << p.next_mark << ' ' << p.max_cycles << ' ' << p.max_residual << ' '
         << p.first_step << ' ' << p.last_step << ' ' << p.step;
    for (double b : p.base)
      file << ' ' << b;
    for (int m = 0; m < p.next_mark; ++m) {
      const auto &w = p.marks[m];
      file << '\n' << w.time << ' ' << w.friction;
      for (int k = 0; k < 3; ++k)
        file << ' ' << w.impulse[k] << ' ' << w.angular[k];
      for (int k = 0; k < wind3d::kComponents; ++k)
        file << ' ' << w.drag[k] << ' ' << w.lift[k];
    }
    file << '\n';
    if (!file)
      return false;
  }
  return std::rename(temporary.c_str(), path.c_str()) == 0;
}

bool LoadProgress(const std::string &path, Progress &p) {
  std::ifstream file(path);
  if (!(file >> p.index >> p.switch_time >> p.next_mark >> p.max_cycles >> p.max_residual >> p.first_step >>
        p.last_step >> p.step))
    return false;
  for (double &b : p.base)
    file >> b;
  if (p.next_mark < 0 || p.next_mark > quarters + 1)
    return false;
  for (int m = 0; m < p.next_mark; ++m) {
    auto &w = p.marks[m];
    file >> w.time >> w.friction;
    for (int k = 0; k < 3; ++k)
      file >> w.impulse[k] >> w.angular[k];
    for (int k = 0; k < wind3d::kComponents; ++k)
      file >> w.drag[k] >> w.lift[k];
  }
  return bool(file);
}

// Step number stored in a solver checkpoint header, or 0 if the file is
// missing. The header starts with an 8-byte magic and three ints; the 64-bit
// step follows at its 8-byte alignment, offset 24.
uint64_t CheckpointStep(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  char head[32];
  uint64_t step = 0;
  if (file.read(head, sizeof(head)) && std::string(head, 8) == "W3DCKPT1")
    std::memcpy(&step, head + 24, sizeof(step));
  return step;
}

bool Same(const Controls &a, const Controls &b) {
  return a.speed == b.speed && wind3d::SameAircraftGeometry(a, b);
}

void Usage() {
  std::cout << "wind3d_sweep [--grid 64.." << wind3d::max_grid
            << " | --grid max] [--device N] [--gpus 1..8] [--speed 1..100] [--angle DEG]\n"
               "  [--preset controls|polar|cruise | --cases FILE] [--output results.csv]\n"
               "  [--settle-time S] [--case-time S] [--average-time S] [--viscosity M2S]\n"
               "  [--wall-model | --laminar] [--wall-viscosity M2S] [--cfl C] [--friction-only]\n"
               "  [--sponge METRES (default 2, 0 disables)] [--cut-cells (with --friction-only)]\n"
               "  [--cubic] [--first-order]\n"
               "  [--pressure-tolerance T] [--pressure-cycles N] [--dump-dir DIR]\n"
               "  [--checkpoint DIR [--checkpoint-minutes M]] (resumes from DIR if it holds a checkpoint)\n"
               "Cases file rows: name angle flap aileron elevator rudder ('#' starts a comment).\n";
}
}  // namespace

int main(int argc, char **argv) {
  int device = 0, grid = 384;
  bool max_grid = false;
  float settle_time = 1.0f, case_time = 0.5f, average_time = 0.25f;
  std::string preset = "controls", cases_path, output = "wind3d_sweep.csv", dump, checkpoint;
  float checkpoint_minutes = 60;
  Controls base;
  base.speed = 30;
  base.angle = 2;
  wind3d::SolverOptions solver;
  solver.display = false;
  solver.viscosity = 1.5e-5f;
  solver.wall_model = true;
  solver.sponge = 2;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto value = [&]() -> std::string {
      if (++i >= argc) {
        std::cerr << "Missing value for " << key << '\n';
        std::exit(1);
      }
      return argv[i];
    };
    auto number = [&]() {
      std::string text = value();
      char *end = nullptr;
      float result = std::strtof(text.c_str(), &end);
      if (text.empty() || *end || !std::isfinite(result)) {
        std::cerr << "Invalid number for " << key << '\n';
        std::exit(1);
      }
      return result;
    };
    if (key == "--help") {
      Usage();
      return 0;
    } else if (key == "--grid") {
      std::string text = value();
      if (text == "max")
        max_grid = true;
      else
        grid = std::atoi(text.c_str());
    } else if (key == "--device")
      device = int(number());
    else if (key == "--gpus")
      solver.devices = int(number());
    else if (key == "--speed")
      base.speed = number();
    else if (key == "--angle")
      base.angle = number();
    else if (key == "--preset")
      preset = value();
    else if (key == "--cases")
      cases_path = value();
    else if (key == "--output")
      output = value();
    else if (key == "--checkpoint")
      checkpoint = value();
    else if (key == "--checkpoint-minutes")
      checkpoint_minutes = number();
    else if (key == "--dump-dir")
      dump = value();
    else if (key == "--settle-time")
      settle_time = number();
    else if (key == "--case-time")
      case_time = number();
    else if (key == "--average-time")
      average_time = number();
    else if (key == "--viscosity")
      solver.viscosity = number();
    else if (key == "--wall-viscosity")
      solver.wall_viscosity = number();
    else if (key == "--wall-model")
      solver.wall_model = true;
    else if (key == "--laminar")
      solver.wall_model = false;
    else if (key == "--cubic")
      solver.cubic = true;
    else if (key == "--cut-cells")
      solver.cut_cells = true;
    else if (key == "--sponge")
      solver.sponge = number();
    else if (key == "--friction-only")
      solver.wall_momentum = false;
    else if (key == "--cfl")
      solver.cfl = number();
    else if (key == "--first-order")
      solver.corrected_transport = false;
    else if (key == "--pressure-tolerance")
      solver.pressure_tolerance = number();
    else if (key == "--pressure-cycles")
      solver.pressure_cycles = int(number());
    else {
      std::cerr << "Unknown option: " << key << '\n';
      Usage();
      return 1;
    }
  }
  if (base.speed < 1 || base.speed > 100 || !(average_time > 0) || average_time > case_time || settle_time < 0) {
    std::cerr << "Invalid speed or timing\n";
    return 1;
  }
  std::vector<Case> cases;
  if (!cases_path.empty() ? !ParseCases(cases_path, base, cases) : (cases = Preset(preset, base)).empty()) {
    std::cerr << "No cases\n";
    return 1;
  }
  for (const auto &c : cases)
    if (c.controls.angle < -20 || c.controls.angle > 25 || c.controls.flap < 0 || c.controls.flap > 40 ||
        std::abs(c.controls.aileron) > 25 || std::abs(c.controls.elevator) > 25 || std::abs(c.controls.rudder) > 25) {
      std::cerr << "Case " << c.name << " is outside the control limits\n";
      return 1;
    }
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    std::cerr << "CUDA device " << device << " is unavailable\n";
    return 1;
  }
  if (max_grid)
    grid = LargestGrid(device);
  solver.display = !dump.empty();
  if (!wind3d::ValidateGeometry() || !wind3d::ValidateWallDiffusion(device) ||
      !wind3d::ValidateClosedSurface(device, false) || !wind3d::ValidateClosedSurface(device, true))
    return 2;
  auto simulation = wind3d::Simulation::Create(device, grid, solver);
  if (!simulation) {
    std::cerr << "CUDA initialization failed for grid " << grid << '\n';
    return 1;
  }
  const auto reference = wind3d::MainWingReference(base);
  const float h = 16.0f / grid;
  std::cout << "device=" << properties.name << " grid=" << grid << 'x' << grid / 2 << 'x' << grid * 3 / 4 << " h=" << h
            << " m solver_gb=" << wind3d::SolverBytes(grid) / 1e9 << " speed=" << base.speed
            << " cubic=" << solver.cubic << " cfl=" << solver.cfl << " sponge=" << solver.sponge
            << " cut_cells=" << solver.cut_cells << " viscosity=" << solver.viscosity << " wall_model="
            << (solver.wall_model ? (solver.wall_momentum ? "log-law" : "log-law-loads-only") : "laminar")
            << " S=" << reference.area << " b=" << reference.span << " mac=" << reference.chord << std::endl;
  Progress saved;
  const std::string fields_path = checkpoint + "/fields.bin", progress_path = checkpoint + "/progress.txt";
  bool resume = !checkpoint.empty() && LoadProgress(progress_path, saved) && saved.step &&
                CheckpointStep(fields_path) == saved.step;
  if (resume && saved.index >= cases.size()) {
    std::cout << "All cases already completed in " << output << std::endl;
    return 0;
  }
  if (resume)
    std::cout << "resume case=" << cases[saved.index].name << " step=" << saved.step << std::endl;
  // A run killed before its first case leaves an empty file; resume then needs
  // the header too.
  bool header = !resume;
  if (resume) {
    std::ifstream existing(output);
    header = existing.peek() == std::ifstream::traits_type::eof();
  }
  std::ofstream csv(output, resume ? std::ios::app : std::ios::trunc);
  if (!csv) {
    std::cerr << "Cannot write " << output << '\n';
    return 1;
  }
  if (header)
    csv << "case,angle,flap,aileron,elevator,rudder,speed,grid,h,viscosity,wall_model,t_start,t_end,steps,Fx,Fy,Fz,Mx,"
           "My,Mz,CL,CD,CY,Cl,Cm,Cn,LD,CL_spread,CD_spread,dCL,dCD,dCY,dCl,dCm,dCn,max_pressure_cycles,max_residual,"
           "wall_seconds,CDf,CDp,CD_fuselage,CD_wing,CD_htail,CD_vtail,CD_nacelle,CL_fuselage,CL_wing,CL_htail,"
           "CL_vtail,CL_nacelle\n";
  // Flushed now: a preempted job is killed without flushing its buffers.
  csv.flush();

  const double q = 0.5 * 1.225 * double(base.speed) * base.speed;
  double base_coefficients[6]{};
  auto start_clock = std::chrono::steady_clock::now(), last_checkpoint = start_clock;
  const size_t first_index = resume ? saved.index : 0;
  if (resume) {
    std::copy(saved.base, saved.base + 6, base_coefficients);
    simulation->Restore(fields_path);
  }
  simulation->Configure(cases[first_index].controls, false);
  simulation->Reset();
  uint64_t generation_seen = 0, checkpoint_step = resume ? saved.step : 0;
  for (size_t index = first_index; index < cases.size(); ++index) {
    const auto &c = cases[index];
    auto case_clock = std::chrono::steady_clock::now();
    simulation->Configure(c.controls, false);
    double switch_time = -1;
    float wait = index ? case_time : settle_time + case_time;
    Window marks[quarters + 1];
    int next_mark = 0, max_cycles = 0, fast_steps = 0;
    float max_residual = 0;
    uint64_t first_step = 0, last_step = 0, previous_step = ~uint64_t(0);
    if (resume && index == saved.index) {
      switch_time = saved.switch_time;
      next_mark = saved.next_mark;
      max_cycles = saved.max_cycles;
      max_residual = saved.max_residual;
      first_step = saved.first_step;
      last_step = saved.last_step;
      std::copy(saved.marks, saved.marks + quarters + 1, marks);
      resume = false;
    }
    auto progress = [&](uint64_t step, size_t at) {
      Progress p;
      p.index = at;
      p.step = step;
      std::copy(base_coefficients, base_coefficients + 6, p.base);
      if (at == index) {
        p.switch_time = switch_time;
        p.next_mark = next_mark;
        p.max_cycles = max_cycles;
        p.max_residual = max_residual;
        p.first_step = first_step;
        p.last_step = last_step;
        std::copy(marks, marks + quarters + 1, p.marks);
      }
      return p;
    };
    auto last_report = std::chrono::steady_clock::now();
    while (next_mark <= quarters) {
      if (!checkpoint.empty() &&
          std::chrono::steady_clock::now() - last_checkpoint > std::chrono::duration<float>(60 * checkpoint_minutes)) {
        last_checkpoint = std::chrono::steady_clock::now();
        // Taken before the request: every recorded mark predates the
        // checkpointed step, and later marks are retaken after a resume.
        Progress p = progress(0, index);
        if (uint64_t step = simulation->Checkpoint(fields_path)) {
          p.step = checkpoint_step = step;
          bool saved_progress = SaveProgress(progress_path, p);
          std::cout << "checkpoint step=" << step << (saved_progress ? "" : " (progress not saved)") << std::endl;
        } else {
          std::cerr << "Checkpoint failed\n";
        }
      }
      if (!simulation->Error().empty()) {
        std::cerr << "Solver error: " << simulation->Error() << '\n';
        return 1;
      }
      auto state = simulation->Latest();
      if (!state || state->step == previous_step || !Same(state->controls, c.controls)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      previous_step = state->step;
      generation_seen = state->generation;
      if (switch_time < 0) {
        switch_time = state->time - state->dt;
        first_step = state->step;
      }
      double begin = switch_time + wait - average_time;
      if (state->time >= begin) {
        max_cycles = std::max(max_cycles, state->pressure_cycles);
        max_residual = std::max(max_residual, state->pressure_residual);
      }
      while (next_mark <= quarters && state->time >= begin + average_time * next_mark / quarters) {
        marks[next_mark++] = Mark(*state);
        last_step = state->step;
      }
      if (std::chrono::steady_clock::now() - last_report > std::chrono::seconds(60)) {
        last_report = std::chrono::steady_clock::now();
        std::cout << "  " << c.name << " t=" << state->time << " / " << switch_time + wait << " step=" << state->step
                  << " step_s=" << state->step_seconds << " cycles=" << state->pressure_cycles
                  << " Fx=" << state->force.x << " Fy=" << state->force.y << " max_speed=" << state->max_speed << " at "
                  << state->fastest.x << ',' << state->fastest.y << ',' << state->fastest.z << std::endl;
      }
      // A control change or impulsive start can briefly drive a fast jet through
      // a narrow slot; only a persistent or extreme excess counts as unstable.
      fast_steps = state->max_speed > 5 * base.speed ? fast_steps + 1 : 0;
      if (fast_steps > 200 || state->max_speed > 20 * base.speed) {
        std::cerr << "Unstable flow in " << c.name << ": speed " << state->max_speed << " m/s at " << state->fastest.x
                  << ',' << state->fastest.y << ',' << state->fastest.z << " t=" << state->time
                  << " step=" << state->step << '\n';
        return 3;
      }
    }
    (void)generation_seen;
    double duration = marks[quarters].time - marks[0].time, force[3], moment[3];
    for (int k = 0; k < 3; ++k) {
      force[k] = (marks[quarters].impulse[k] - marks[0].impulse[k]) / duration;
      moment[k] = (marks[quarters].angular[k] - marks[0].angular[k]) / duration;
    }
    double low_cl = 1e30, high_cl = -1e30, low_cd = 1e30, high_cd = -1e30;
    for (int k = 0; k < quarters; ++k) {
      double span = marks[k + 1].time - marks[k].time;
      double cl = (marks[k + 1].impulse[1] - marks[k].impulse[1]) / span / (q * reference.area);
      double cd = (marks[k + 1].impulse[0] - marks[k].impulse[0]) / span / (q * reference.area);
      low_cl = std::min(low_cl, cl);
      high_cl = std::max(high_cl, cl);
      low_cd = std::min(low_cd, cd);
      high_cd = std::max(high_cd, cd);
    }
    // Wind axes: freestream +x, so drag is +x, lift +y and side force +z.
    // Roll, yaw and pitch moments are about world x, y and z at the body origin.
    double coefficients[6] = {force[1] / (q * reference.area),
                              force[0] / (q * reference.area),
                              force[2] / (q * reference.area),
                              moment[0] / (q * reference.area * reference.span),
                              moment[2] / (q * reference.area * reference.chord),
                              moment[1] / (q * reference.area * reference.span)};
    if (index == 0)
      std::copy(coefficients, coefficients + 6, base_coefficients);
    double friction_drag = (marks[quarters].friction - marks[0].friction) / duration / (q * reference.area);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - case_clock).count();
    char line[1024];
    std::snprintf(line, sizeof(line),
                  "%s,%g,%g,%g,%g,%g,%g,%d,%.6g,%.3g,%s,%.5f,%.5f,%llu,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6f,%.6f,%.6f,"
                  "%.6f,%.6f,%.6f,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.3g,%.1f,%.6f,%.6f",
                  c.name.c_str(), c.controls.angle, c.controls.flap, c.controls.aileron, c.controls.elevator,
                  c.controls.rudder, base.speed, grid, h, solver.viscosity,
                  !solver.wall_model     ? "laminar"
                  : solver.wall_momentum ? "log-law"
                                         : "log-law-loads",
                  marks[0].time, marks[quarters].time, (unsigned long long)(last_step - first_step + 1), force[0],
                  force[1], force[2], moment[0], moment[1], moment[2], coefficients[0], coefficients[1],
                  coefficients[2], coefficients[3], coefficients[4], coefficients[5], coefficients[0] / coefficients[1],
                  high_cl - low_cl, high_cd - low_cd, coefficients[0] - base_coefficients[0],
                  coefficients[1] - base_coefficients[1], coefficients[2] - base_coefficients[2],
                  coefficients[3] - base_coefficients[3], coefficients[4] - base_coefficients[4],
                  coefficients[5] - base_coefficients[5], max_cycles, max_residual, seconds, friction_drag,
                  coefficients[1] - friction_drag);
    if (!dump.empty()) {
      // Downsampled cell-centred u, v, w, p of the last snapshot, for inspection.
      auto state = simulation->Latest();
      std::ofstream file(dump + "/" + c.name + ".field", std::ios::binary);
      int header[3] = {state->nx, state->ny, state->nz};
      file.write(reinterpret_cast<const char *>(header), sizeof(header));
      file.write(reinterpret_cast<const char *>(state->cells.data()), state->cells.size() * sizeof(wind3d::Cell));
    }
    std::string breakdown;
    for (int pass = 0; pass < 2; ++pass)
      for (int k = 0; k < wind3d::kComponents; ++k) {
        const double *a = pass ? marks[0].lift : marks[0].drag, *b = pass ? marks[quarters].lift : marks[quarters].drag;
        char item[32];
        std::snprintf(item, sizeof(item), ",%.6f", (b[k] - a[k]) / duration / (q * reference.area));
        breakdown += item;
      }
    csv << line << breakdown << '\n';
    csv.flush();
    if (checkpoint_step)
      SaveProgress(progress_path, progress(checkpoint_step, index + 1));
    std::cout << "case " << line << breakdown << std::endl;
  }
  std::cout << "total_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - start_clock).count()
            << std::endl;
  return 0;
}
