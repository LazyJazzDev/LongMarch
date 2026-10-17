#include <fmt/format.h>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "grassland/rhi/rhi.h"
#include "grassland/util/fps_counter.h"

namespace {
int RunWindow(grassland::rhi::IDevice *device, grassland::rhi::IWindow *window, grassland::rhi::IQueue *graphics) {
  using namespace grassland::rhi;
  const auto options = window->GetSupportedPresentationOptions();
  if (options.empty())
    return 1;
  std::cout << "Supported presentation combinations (SDR):\n";
  for (const auto &option : options) {
    std::cout << "  " << PresentModeName(option.present_mode) << " + " << ImageFormatName(option.format) << '\n';
  }
  bool previous_p = false;
  PresentMode requested = PRESENT_MODE_VSYNC;
  grassland::FPSCounter fps_counter;
  auto title_time = std::chrono::steady_clock::now();
  while (true) {
    if (!PollEvents())
      break;
    if (window->ShouldClose())
      break;
    if (window->IsKeyDown(GLFW_KEY_ESCAPE))
      break;
    // Previous loop locals are gone; rebuild only before recording this frame.
    if (window->IsSwapChainInvalid() && !window->RebuildSwapChain())
      return 1;
    const bool p = window->IsKeyDown(GLFW_KEY_P);
    if (p && !previous_p) {
      requested = requested == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
      if (!window->SetPresentMode(requested))
        return 1;
      std::cout << "Present mode: " << PresentModeName(window->GetPresentMode()) << std::endl;
    }
    previous_p = p;
    ObjPtr<IImage> image = window->GetNextImage();
    if (!image) {
      fps_counter.Reset();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    ObjPtr<ICommandContext> context = device->CreateCommandContext();
    if (!context)
      return 1;
    if (!context->CmdClearColor(image.Get(), {0.6f, 0.7f, 0.8f, 1.0f}))
      return 1;
    if (!context->CmdPresent(window))
      return 1;
    // Submit ends recording and borrows the caller's context reference.
    ObjPtr<ISubmissionToken> token = graphics->Submit(context.Get());
    if (!token)
      return 1;
    // Local containers release caller references; accepted work retains its resources.
    fps_counter.TickFrame();
    const auto now = std::chrono::steady_clock::now();
    if (now - title_time >= std::chrono::milliseconds(250)) {
      if (!window->SetTitle(fmt::format("LongMarch RHI - FPS: {:.1f} - P: toggle VSync", fps_counter.GetFPS())))
        return 1;
      title_time = now;
    }
  }
  return 0;
}
}  // namespace

int main(int argc, char **argv) {
  using namespace grassland::rhi;
  BackendAPI api = BACKEND_API_DEFAULT;
  if (argc > 2) {
    std::cerr << "Usage: demo_rhi_hello [d3d12|vulkan|metal]\n";
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
      std::cerr << "Usage: demo_rhi_hello [d3d12|vulkan|metal]\n";
      return 1;
    }
  }

  ObjPtr<ICore> core = CreateCore(api);
  if (!core)
    return 1;
  const auto devices = core->GetPhysicalDevices();
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
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection(requirements);
  // For an explicit ID, use core->CreateDevice(id) instead.
  if (!device)
    return 1;
  std::cout << "Selected: " << device->Name() << '\n'
            << "Enabled: ray tracing pipeline=" << device->SupportRayTracingPipeline()
            << ", ray query=" << device->SupportRayQuery() << ", geometry shader=" << device->SupportGeometryShader()
            << '\n';
  ObjPtr<IWindow> window = device->CreateGLFWWindow({1280, 720, "LongMarch RHI - P: toggle VSync"});
  if (!window)
    return 1;

  ObjPtr<IQueue> graphics = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!graphics)
    return 1;
  // Reverse declaration order releases children before their parents on every return.
  return RunWindow(device.Get(), window.Get(), graphics.Get());
}
