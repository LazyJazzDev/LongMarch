#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "render.h"
#include "validation.h"

int main(int argc, char **argv) {
  using namespace grassland::rhi;
  int grid = 192, steps = 20, frames = 0;
  float validate_time = 0;
  bool validate = false, headless = false, debug = false;
  BackendAPI backend = BACKEND_API_VULKAN;
  wind3d::Controls controls;
  wind3d::ViewSettings view;
  wind3d::SolverOptions solver;
  std::string capture;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--validate")
      validate = true;
    else if (key == "--first-order")
      solver.corrected_transport = false;
    else if (key == "--no-flow")
      view.flow = false;
    else if (key == "--model-only")
      view.flow = view.loads = false;
    else if (key == "--headless")
      headless = true;
    else if (key == "--debug")
      debug = true;
    else if (key == "--help") {
      std::cout
          << "demo_wind3d [--backend vulkan|d3d12] [--grid 64..512, multiple of 16] "
             "[--speed 1..30] [--angle -20..25] [--flap 0..40] [--aileron -25..25] "
             "[--elevator -25..25] [--rudder -25..25] [--frames N] [--headless] "
             "[--first-order] [--validate-time SECONDS] [--pressure-tolerance 1e-6..0.1] [--pressure-cycles 1..256] [--model-only] [--yaw RADIANS] [--pitch RADIANS] [--distance METRES] [--capture image.ppm] [--validate --steps N] [--debug]\n";
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
      else {
        char *end = nullptr;
        float number = std::strtof(value.c_str(), &end);
        if (value.empty() || *end || !std::isfinite(number)) {
          std::cerr << "Invalid number\n";
          return 1;
        }
        if (key == "--grid" || key == "--steps" || key == "--frames" || key == "--pressure-cycles") {
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
        } else if (key == "--validate-time") {
          if (number <= 0)
            return 1;
          validate_time = number;
          validate = true;
        } else if (key == "--pressure-tolerance")
          solver.pressure_tolerance = number;
        else if (key == "--yaw")
          view.yaw = number;
        else if (key == "--pitch")
          view.pitch = number;
        else if (key == "--distance")
          view.distance = number;
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
  if (grid < 64 || grid > 512 || grid % 16 || steps < 1 || controls.speed < 1 || controls.speed > 30 ||
      controls.angle < -20 || controls.angle > 25 || controls.flap < 0 || controls.flap > 40 ||
      std::abs(view.pitch) > 1.55f || view.distance < 7 || view.distance > 25 || std::abs(controls.aileron) > 25 ||
      std::abs(controls.elevator) > 25 || std::abs(controls.rudder) > 25)
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
  auto simulation = wind3d::Simulation::Create(device->GetCUDADeviceIndex(), grid, solver);
  if (!simulation) {
    std::cerr << "CUDA initialization failed\n";
    return 1;
  }
  simulation->Configure(controls, false);
  simulation->Reset();
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
    if (state && state->generation >= 1 && state->step != previous) {
      previous = state->step;
      std::cout << "step=" << state->step << " time=" << state->time << " seconds=" << state->step_seconds
                << " pressure_cycles=" << state->pressure_cycles << " residual=" << state->pressure_residual
                << " div=" << state->divergence_before << " -> " << state->divergence_after
                << " force=" << state->force.x << ',' << state->force.y << ',' << state->force.z
                << " moment=" << state->moment.x << ',' << state->moment.y << ',' << state->moment.z << std::endl;
      passed &= state->divergence_after < std::max(1e-5f, state->divergence_before * solver.pressure_tolerance * 2);
      passed &= state->pressure_residual <= solver.pressure_tolerance;
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
