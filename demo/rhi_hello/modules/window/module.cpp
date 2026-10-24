#include "../../module.h"

#include <fmt/format.h>

#include <array>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "grassland/rhi/rhi.h"
#include "grassland/util/fps_counter.h"

namespace {
int RunWindow(grassland::rhi::IDevice *device,
              grassland::rhi::IWindow *window,
              grassland::rhi::IQueue *graphics,
              int frames,
              bool headless,
              grassland::rhi::PresentMode &requested,
              bool &present_mode_changed) {
  using namespace grassland::rhi;
  if (!headless) {
    const auto options = window->GetSupportedPresentationOptions();
    if (options.empty())
      return 1;
    std::cout << "Supported presentation combinations (SDR):\n";
    for (const auto &option : options) {
      std::cout << "  " << PresentModeName(option.present_mode) << " + " << ImageFormatName(option.format) << '\n';
    }
  }
  std::array<ObjPtr<IImage>, 3> targets;
  std::array<ObjPtr<ISubmissionToken>, 3> pending;
  if (headless) {
    for (auto &target : targets) {
      target = device->CreateImage({1280, 720, IMAGE_FORMAT_RGBA8_UNORM});
      if (!target)
        return 1;
    }
  }
  grassland::FPSCounter fps_counter;
  auto title_time = std::chrono::steady_clock::now();
  // Warm up allocations and pipelines before measuring completed GPU work.
  auto benchmark_start = std::chrono::steady_clock::now();
  int rendered = headless ? -60 : 0;
  while (!frames || rendered < frames) {
    if (!headless) {
      if (!PollEvents())
        break;
      if (window->ShouldClose())
        break;
      // Previous loop locals are gone; rebuild only before recording this frame.
      if (window->IsSwapChainInvalid()) {
        if (!window->RebuildSwapChain())
          return 1;
      }
      if (present_mode_changed) {
        present_mode_changed = false;
        if (!window->SetPresentMode(requested))
          return 1;
        std::cout << "Present mode: " << PresentModeName(window->GetPresentMode()) << std::endl;
      }
    }
    const size_t slot = (static_cast<int64_t>(rendered) + 60) % targets.size();
    if (headless) {
      if (rendered == 0) {
        for (auto &token : pending) {
          if (token) {
            if (!token->WaitHost())
              return 1;
            token = {};
          }
        }
        benchmark_start = std::chrono::steady_clock::now();
      }
      if (pending[slot]) {
        if (!pending[slot]->WaitHost())
          return 1;
        pending[slot] = {};
      }
    }
    ObjPtr<IImage> image;
    if (headless)
      image = targets[slot];
    else
      image = window->GetNextImage();
    if (!image) {
      if (headless)
        return 1;
      fps_counter.Reset();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    ObjPtr<ICommandContext> context = device->CreateCommandContext();
    if (!context)
      return 1;
    if (!context->CmdClearColor(image.Get(), {0.6f, 0.7f, 0.8f, 1.0f}))
      return 1;
    if (!headless) {
      if (!context->CmdPresent(window))
        return 1;
    }
    // Submit ends recording and borrows the caller's context reference.
    ObjPtr<ISubmissionToken> token = graphics->Submit(context.Get());
    if (!token)
      return 1;
    // Local containers release caller references; accepted work retains its resources.
    if (headless)
      pending[slot] = std::move(token);
    ++rendered;
    fps_counter.TickFrame();
    const auto now = std::chrono::steady_clock::now();
    if (!headless && now - title_time >= std::chrono::milliseconds(250)) {
      if (!window->SetTitle(fmt::format("LongMarch RHI - FPS: {:.1f} - P: toggle VSync", fps_counter.GetFPS())))
        return 1;
      title_time = now;
    }
  }
  if (headless) {
    for (const auto &token : pending) {
      if (token) {
        if (!token->WaitHost())
          return 1;
      }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - benchmark_start).count();
    std::cout << fmt::format("window: headless 1280x720, {} frames, {:.6f} s, {:.2f} FPS (3 in flight, 60 warmup)\n",
                             rendered, seconds, rendered / seconds);
  }
  return 0;
}
}  // namespace

int RunWindow(grassland::rhi::BackendAPI api, int frames, bool headless) {
  using namespace grassland::rhi;
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
  PresentMode requested = PRESENT_MODE_VSYNC;
  bool present_mode_changed = false;
  ObjPtr<IWindow> window;
  if (!headless) {
    window = device->CreateGLFWWindow({1280, 720, "LongMarch RHI - P: toggle VSync"});
    if (!window)
      return 1;
  }

  ObjPtr<IQueue> graphics = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!graphics)
    return 1;
  if (!headless) {
    window->KeyEvent().RegisterCallback([&](int key, int, int action, int) {
      if (action != GLFW_PRESS)
        return;
      if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window->GLFWWindow(), GLFW_TRUE);
      if (key == GLFW_KEY_P) {
        requested = requested == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
        present_mode_changed = true;
      }
    });
  }
  // Reverse declaration order releases children before their parents on every return.
  return RunWindow(device.Get(), window.Get(), graphics.Get(), frames, headless, requested, present_mode_changed);
}
