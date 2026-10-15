#include <fmt/format.h>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "grassland/rhi/rhix/rhix.h"
#include "grassland/util/fps_counter.h"

int main(int argc, char **argv) {
  using namespace grassland::rhi;
  using namespace grassland::rhix;
  BackendAPI api = BACKEND_API_DEFAULT;
  if (argc > 2) {
    std::cerr << "Usage: demo_rhix_hello [d3d12|vulkan|metal]\n";
    return 1;
  }
  if (argc == 2) {
    const std::string name = argv[1];
    if (name == "d3d12")
      api = BACKEND_API_D3D12;
    else if (name == "vulkan")
      api = BACKEND_API_VULKAN;
    else if (name == "metal")
      api = BACKEND_API_METAL;
    else {
      std::cerr << "Usage: demo_rhix_hello [d3d12|vulkan|metal]\n";
      return 1;
    }
  }

  auto core = grassland::rhix::CreateCore(api);
  if (!core)
    return 1;
  const auto devices = core.GetPhysicalDevices();
  std::cout << "Backend: " << BackendAPIName(api) << '\n' << std::boolalpha;
  for (size_t id = 0; id < devices.size(); ++id) {
    const auto &info = devices[id];
    std::cout << '[' << id << "] " << info.name << " (" << PhysicalDeviceTypeName(info.type) << ")\n"
              << "  Dedicated: " << info.dedicated_memory / (1024 * 1024)
              << " MiB, shared/unified: " << info.unified_memory / (1024 * 1024) << " MiB\n"
              << "  Ray tracing pipeline: " << info.ray_tracing_pipeline_feature
              << ", ray query: " << info.ray_query_feature << ", geometry shader: " << info.geometry_shader_feature
              << '\n';
  }

  // Requirements only filter candidates. All supported RHI features are enabled.
  DeviceFeatureRequirement requirements{};
  auto device = core.CreateDeviceByAutoSelection(requirements);
  // For an explicit ID, use core.CreateDevice(id) instead.
  if (!device)
    return 1;
  std::cout << "Selected: " << device.Name() << '\n'
            << "Enabled: ray tracing pipeline=" << device.SupportRayTracingPipeline()
            << ", ray query=" << device.SupportRayQuery() << ", geometry shader=" << device.SupportGeometryShader()
            << '\n';
  if (api == BACKEND_API_METAL)
    return 0;  // Metal keeps the device-query demo until its presentation backend is added.
  auto window = device.CreateGLFWWindow({1280, 720, "LongMarch RHIX - P: toggle VSync"});
  auto graphics = device.GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!window || !graphics)
    return 1;
  const auto options = window.GetSupportedPresentationOptions();
  if (options.empty())
    return 1;
  std::cout << "Supported presentation combinations (SDR):\n";
  for (const auto &option : options) {
    const char *format = option.format == IMAGE_FORMAT_RGBA8_UNORM ? "RGBA8 UNORM" : "BGRA8 UNORM";
    std::cout << "  " << (option.present_mode == PRESENT_MODE_VSYNC ? "VSync" : "Continuous") << " + " << format
              << '\n';
  }
  bool previous_p = false;
  PresentMode requested = PRESENT_MODE_VSYNC;
  grassland::FPSCounter fps_counter;
  auto title_time = std::chrono::steady_clock::now();
  while (Window::PollEvents() && !window.ShouldClose() && !window.IsKeyDown(KEY_ESCAPE)) {
    const bool p = window.IsKeyDown(KEY_P);
    if (p && !previous_p) {
      requested = requested == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
      if (!window.SetPresentMode(requested))
        return 1;
      std::cout << "Present mode: " << (window.GetPresentMode() == PRESENT_MODE_VSYNC ? "VSync" : "Continuous")
                << std::endl;
    }
    previous_p = p;
    auto image = window.AcquireFrame();
    if (!image) {
      fps_counter.Reset();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    auto context = device.CreateCommandContext(graphics);
    if (!context || !context.ClearColor(image, {0.6f, 0.7f, 0.8f, 1.0f}) || !context.Present(image) ||
        !graphics.Submit(context.Finish()))
      return 1;
    fps_counter.TickFrame();
    const auto now = std::chrono::steady_clock::now();
    if (now - title_time >= std::chrono::milliseconds(250)) {
      if (!window.SetTitle(fmt::format("LongMarch RHIX - FPS: {:.1f} - P: toggle VSync", fps_counter.GetFPS())))
        return 1;
      title_time = now;
    }
  }
  // Explicit child-before-parent release; Device drains remaining GPU work.
  graphics.Release();
  window.Release();
  device.Release();
  core.Release();
}
