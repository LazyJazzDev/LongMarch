#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "render.h"
#include "validation.h"

int main(int argc, char **argv) {
  using namespace grassland::rhi;
  int grid = 192, steps = 20, frames = 0;
  bool validate = false, headless = false, debug = false;
  BackendAPI backend = BACKEND_API_VULKAN;
  wind3d::Controls controls;
  std::string capture;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--validate")
      validate = true;
    else if (key == "--headless")
      headless = true;
    else if (key == "--debug")
      debug = true;
    else if (key == "--help") {
      std::cout << "demo_wind3d [--backend vulkan|d3d12] [--grid 64..256, multiple of 16] "
                   "[--speed 1..30] [--angle -20..25] [--flap 0..40] [--aileron -25..25] "
                   "[--elevator -25..25] [--rudder -25..25] [--frames N] [--headless] "
                   "[--capture image.ppm] [--validate --steps N] [--debug]\n";
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
        if (key == "--grid" || key == "--steps" || key == "--frames") {
          if (number != std::floor(number) || number < 0 || number > 1000000)
            return 1;
          if (key == "--grid")
            grid = int(number);
          if (key == "--steps")
            steps = int(number);
          if (key == "--frames")
            frames = int(number);
        } else if (key == "--speed")
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
  if (grid < 64 || grid > 256 || grid % 16 || steps < 1 || controls.speed < 1 || controls.speed > 30 ||
      controls.angle < -20 || controls.angle > 25 || controls.flap < 0 || controls.flap > 40 ||
      std::abs(controls.aileron) > 25 || std::abs(controls.elevator) > 25 || std::abs(controls.rudder) > 25)
    return 1;
  if (headless && !frames)
    frames = 300;
  if (!capture.empty() && !frames) {
    std::cerr << "Capture requires a frame limit\n";
    return 1;
  }
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
  auto simulation = wind3d::Simulation::Create(device->GetCUDADeviceIndex(), grid);
  if (!simulation) {
    std::cerr << "CUDA initialization failed\n";
    return 1;
  }
  simulation->Configure(controls, false);
  simulation->Reset();
  if (!validate)
    return wind3d::Render(device.Get(), *simulation, controls, frames, headless, capture) ? 0 : 1;
  uint64_t previous = 0;
  bool passed = true;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(3);
  while (std::chrono::steady_clock::now() < deadline) {
    if (!simulation->Error().empty()) {
      std::cerr << simulation->Error() << '\n';
      return 1;
    }
    auto state = simulation->Latest();
    if (state && state->generation >= 1 && state->step != previous) {
      previous = state->step;
      std::cout << "step=" << state->step << " time=" << state->time << " seconds=" << state->step_seconds
                << " div=" << state->divergence_before << " -> " << state->divergence_after
                << " force=" << state->force.x << ',' << state->force.y << ',' << state->force.z
                << " moment=" << state->moment.x << ',' << state->moment.y << ',' << state->moment.z << std::endl;
      passed &= state->divergence_after < std::max(0.002f, state->divergence_before * 0.1f);
      if (state->step >= uint64_t(steps)) {
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
