#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "render.h"
#include "validation.h"

int main(int argc, char **argv) {
  using namespace grassland::rhi;
  int grid = 256, steps = 20, frames = 0;
  float validate_time = 0;
  bool validate = false, headless = false, debug = false;
  BackendAPI backend = BACKEND_API_VULKAN;
  wind3d::Controls controls;
  wind3d::ViewSettings view;
  wind3d::SolverOptions solver;
  solver.leapfrog_flow_maps = true;
  solver.export_vorticity = true;
  std::string capture;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--validate")
      validate = true;
    else if (key == "--lfm")
      solver.leapfrog_flow_maps = true;
    else if (key == "--aero") {
      // Reproduce the aerodynamic validation model without changing export,
      // diagnostics, grid or controls selected by other command-line options.
      solver.leapfrog_flow_maps = true;
      solver.viscosity = solver.wall_viscosity = 1.5e-5f;
      solver.wall_model = solver.cut_cells = true;
      solver.wall_momentum = false;
      solver.sponge = 2;
      solver.cfl = 0.25f;
      solver.pressure_cycles = 128;
    } else if (key == "--maccormack")
      solver.leapfrog_flow_maps = false;
    else if (key == "--wall-model")
      solver.wall_model = true;
    else if (key == "--slip-walls") {
      solver.wall_model = true;
      solver.wall_momentum = false;
    } else if (key == "--cut-cells") {
      solver.wall_model = solver.cut_cells = true;
      solver.wall_momentum = false;
    } else if (key == "--cubic")
      solver.cubic = true;
    else if (key == "--no-export")
      solver.export_field = solver.export_vorticity = false;
    else if (key == "--first-order") {
      solver.leapfrog_flow_maps = false;
      solver.corrected_transport = false;
    } else if (key == "--no-flow")
      view.flow = view.volume = false;
    else if (key == "--streamlines")
      view.flow = true;
    else if (key == "--no-volume")
      view.volume = false;
    else if (key == "--model-only")
      view.flow = view.volume = view.loads = false;
    else if (key == "--headless")
      headless = true;
    else if (key == "--debug")
      debug = true;
    else if (key == "--help") {
      std::cout
          << "demo_wind3d [--backend vulkan|d3d12] [--grid 64..640, multiple of 16] "
             "[--speed 1..30] [--angle -20..25] [--flap 0..40] [--aileron -25..25] "
             "[--elevator -25..25] [--rudder -25..25] [--frames N] [--headless] "
             "[--aero] [--lfm|--maccormack|--first-order] [--viscosity NU] [--wall-viscosity NU] [--wall-model|--slip-walls|--cut-cells] [--cubic] [--sponge METRES] [--no-export] [--cfl 0.05..8] [--max-dt SECONDS] [--diagnostics file.csv] [--validate-time SECONDS] [--pressure-tolerance 1e-6..0.1] [--pressure-cycles 1..256] [--model-only] [--yaw RADIANS] [--pitch RADIANS] [--distance METRES] [--capture image.ppm] [--validate --steps N] [--debug]\n"
          << "--aero: LFM, air viscosity 1.5e-5 m^2/s, slip cut cells with wall-shear loads, CFL 0.25, 2 m sponge.\n"
          << "Place --aero before any individual solver overrides.\n";
      std::cout << "--restore FILE resumes a checkpoint; supply its original grid, controls and solver settings.\n";
      std::cout
          << "Flow view: --streamlines --no-volume --streamline-density 1..4 --streamline-width 0.75..4 (pixels).\n";
      std::cout
          << "Airframe: --wing-span 2..4.2 (half-span, m), --wing-camber 0..0.06, --wing-sweep 0..0.7 (slope), --tail-span 0.6..1.4 (half-span, m), --tail-leading 1..2 (body x, m).\n";
      return 0;
    } else {
      if (++i >= argc) {
        std::cerr << "Missing option value\n";
        return 1;
      }
      std::string value = argv[i];
      if (key == "--backend") {
        if (value == "vulkan")
          backend = BACKEND_API_VULKAN;
        else if (value == "d3d12")
          backend = BACKEND_API_D3D12;
        else {
          std::cerr << "Unsupported backend\n";
          return 1;
        }
      } else if (key == "--capture")
        capture = value;
      else if (key == "--diagnostics")
        solver.diagnostics_path = value;
      else if (key == "--restore")
        solver.initial_checkpoint = value;
      else {
        char *end = nullptr;
        float number = std::strtof(value.c_str(), &end);
        if (value.empty() || *end || !std::isfinite(number)) {
          std::cerr << "Invalid number\n";
          return 1;
        }
        if (key == "--grid" || key == "--steps" || key == "--frames" || key == "--pressure-cycles" ||
            key == "--streamline-density") {
          if (number != std::floor(number) || number < 0 || number > 1000000)
            return 1;
          if (key == "--pressure-cycles")
            solver.pressure_cycles = int(number);
          if (key == "--grid")
            grid = int(number);
          if (key == "--steps")
            steps = int(number);
          if (key == "--frames")
            frames = int(number);
          if (key == "--streamline-density")
            view.streamline_density = int(number);
        } else if (key == "--validate-time") {
          if (number <= 0)
            return 1;
          validate_time = number;
          validate = true;
        } else if (key == "--pressure-tolerance")
          solver.pressure_tolerance = number;
        else if (key == "--cfl")
          solver.cfl = number;
        else if (key == "--viscosity")
          solver.viscosity = number;
        else if (key == "--wall-viscosity")
          solver.wall_viscosity = number;
        else if (key == "--sponge")
          solver.sponge = number;
        else if (key == "--max-dt")
          solver.max_dt = number;
        else if (key == "--yaw")
          view.yaw = number;
        else if (key == "--pitch")
          view.pitch = number;
        else if (key == "--distance")
          view.distance = number;
        else if (key == "--streamline-width")
          view.streamline_width = number;
        else if (key == "--wing-span")
          controls.airframe.wing_span = number;
        else if (key == "--wing-camber")
          controls.airframe.wing_camber = number;
        else if (key == "--wing-sweep")
          controls.airframe.wing_sweep = number;
        else if (key == "--tail-span")
          controls.airframe.tail_span = number;
        else if (key == "--tail-leading")
          controls.airframe.tail_leading = number;
        else if (key == "--speed")
          controls.speed = number;
        else if (key == "--angle")
          controls.angle = number;
        else if (key == "--flap")
          controls.flap = number;
        else if (key == "--aileron")
          controls.aileron = number;
        else if (key == "--elevator")
          controls.elevator = number;
        else if (key == "--rudder")
          controls.rudder = number;
        else {
          std::cerr << "Unknown option: " << key << '\n';
          return 1;
        }
      }
    }
  }
  if (view.streamline_density < 1 || view.streamline_density > 4 || view.streamline_width < .75f ||
      view.streamline_width > 4 || !wind3d::ValidAirframe(controls.airframe) || grid < 64 || grid > 640 || grid % 16 ||
      steps < 1 || controls.speed < 1 || controls.speed > 30 || controls.angle < -20 || controls.angle > 25 ||
      controls.flap < 0 || controls.flap > 40 || std::abs(view.pitch) > 1.55f || view.distance < 7 ||
      view.distance > 25 || std::abs(controls.aileron) > 25 || std::abs(controls.elevator) > 25 ||
      std::abs(controls.rudder) > 25)
    return 1;
  if (headless && !frames)
    frames = 300;
  if (!capture.empty() && !frames) {
    std::cerr << "Capture requires a frame limit\n";
    return 1;
  }
  if (validate && !wind3d::ValidateGeometry())
    return 2;
  ObjPtr<ICore> core = CreateCore(backend, {debug});
  if (!core)
    return 1;
  ObjPtr<IDevice> device;
  const auto physical = core->GetPhysicalDevices();
  for (size_t i = 0; i < physical.size(); ++i) {
    if (physical[i].cuda_device < 0)
      continue;
    device = core->CreateDevice(int(i));
    if (device)
      break;
  }
  if (!device) {
    std::cerr << "A CUDA-capable graphics device is required\n";
    return 1;
  }
  if (validate && !wind3d::ValidateWallDiffusion(device->GetCUDADeviceIndex()))
    return 2;
  if (validate && solver.leapfrog_flow_maps && !wind3d::ValidateFlowMaps(device->GetCUDADeviceIndex()))
    return 2;
  if (validate && !wind3d::ValidateClosedSurface(device->GetCUDADeviceIndex(), solver.cut_cells))
    return 2;
  solver.record_loads = !headless && !validate;
  solver.initial_controls = controls;
  auto simulation = wind3d::Simulation::Create(device->GetCUDADeviceIndex(), grid, solver);
  if (!simulation) {
    std::cerr << "CUDA initialization failed\n";
    return 1;
  }
  if (solver.initial_checkpoint.empty()) {
    simulation->Configure(controls, false);
    simulation->Reset();
  }
  if (!validate)
    return wind3d::Render(device.Get(), *simulation, controls, frames, headless, capture, view) ? 0 : 1;
  uint64_t previous = 0;
  bool passed = true;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::hours(6);
  while (std::chrono::steady_clock::now() < deadline) {
    if (!simulation->Error().empty()) {
      std::cerr << simulation->Error() << '\n';
      return 1;
    }
    auto state = simulation->Latest();
    if (state && (solver.initial_checkpoint.size() || state->generation >= 1) && state->step != previous) {
      previous = state->step;
      std::cout << "step=" << state->step << " time=" << state->time << " seconds=" << state->step_seconds
                << " pressure_cycles=" << state->pressure_cycles << " residual=" << state->pressure_residual
                << " div=" << state->divergence_before << " -> " << state->divergence_after
                << " force=" << state->force.x << ',' << state->force.y << ',' << state->force.z
                << " moment=" << state->moment.x << ',' << state->moment.y << ',' << state->moment.z << std::endl;
      passed &= state->divergence_after < std::max(1e-5f, state->divergence_before * solver.pressure_tolerance * 2);
      passed &= state->pressure_residual <= solver.pressure_tolerance;
      if (solver.export_vorticity) {
        passed &= state->vorticity.size() == size_t(grid) * (grid / 2) * (grid * 3 / 4);
        passed &= std::all_of(state->vorticity.begin(), state->vorticity.end(),
                              [](float value) { return std::isfinite(value) && value >= 0; });
      }
      if (validate_time > 0 ? state->time >= validate_time : state->step >= uint64_t(steps)) {
        std::cout << "projection_validation=" << (passed ? "PASS" : "FAIL") << std::endl;
        passed &= wind3d::ValidateLifecycle(*simulation, controls);
        return passed ? 0 : 2;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  std::cerr << "Timed out waiting for CUDA snapshots\n";
  return 1;
}
