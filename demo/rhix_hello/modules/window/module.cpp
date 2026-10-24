#include "../../module.h"

#include <fmt/format.h>

#include <array>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <utility>

#include "../../save_image.h"
#include "grassland/rhi/rhix/rhix.h"
#include "grassland/util/fps_counter.h"

int RunWindow(grassland::rhi::BackendAPI api, int frames, bool headless, const std::string &capture_directory) {
  using namespace grassland::rhi;
  using namespace grassland::rhix;
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
  grassland::rhix::DownloadToken last_frame;
  ImageDesc last_frame_desc{};
  ColorSpace last_frame_space = COLOR_SPACE_SRGB_NONLINEAR;
  float last_frame_white_scale = 1.0f;
  std::cout << "Selected: " << device.Name() << '\n'
            << "Enabled: ray tracing pipeline=" << device.SupportRayTracingPipeline()
            << ", ray query=" << device.SupportRayQuery() << ", geometry shader=" << device.SupportGeometryShader()
            << '\n';
  PresentMode requested = PRESENT_MODE_VSYNC;
  bool present_mode_changed = false;
  grassland::rhix::Window window;
  if (!headless) {
    window = device.CreateGLFWWindow({1280, 720, "LongMarch RHIX - P: toggle VSync"});
    if (!window)
      return 1;
  }

  auto graphics = device.GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!graphics)
    return 1;
  if (!headless) {
    const auto options = window.GetSupportedPresentationOptions();
    if (options.empty())
      return 1;
    std::cout << "Supported presentation combinations (SDR):\n";
    for (const auto &option : options) {
      std::cout << "  " << PresentModeName(option.present_mode) << " + " << ImageFormatName(option.format) << '\n';
    }
  }
  std::array<grassland::rhix::Image, 3> targets;
  std::array<grassland::rhix::SubmissionToken, 3> pending;
  if (headless) {
    for (auto &target : targets) {
      target = device.CreateImage({1280, 720, IMAGE_FORMAT_R8G8B8A8_UNORM});
      if (!target)
        return 1;
    }
  }
  if (!headless) {
    window.KeyEvent().RegisterCallback([&](int key, int, int action, int) {
      if (action != GLFW_PRESS)
        return;
      if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window.GLFWWindow(), GLFW_TRUE);
      if (key == GLFW_KEY_P) {
        requested = requested == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
        present_mode_changed = true;
      }
    });
  }
  grassland::FPSCounter fps_counter;
  auto title_time = std::chrono::steady_clock::now();
  // Warm up allocations and pipelines before measuring completed GPU work.
  auto benchmark_start = std::chrono::steady_clock::now();
  int rendered = headless ? -60 : 0;
  while (!frames || rendered < frames) {
    if (!headless) {
      if (!Window::PollEvents())
        break;
      if (window.ShouldClose())
        break;
      // Previous loop locals are gone; rebuild only before recording this frame.
      if (window.IsSwapChainInvalid()) {
        if (!window.RebuildSwapChain())
          return 1;
      }
      if (present_mode_changed) {
        present_mode_changed = false;
        if (!window.SetPresentMode(requested))
          return 1;
        std::cout << "Present mode: " << PresentModeName(window.GetPresentMode()) << std::endl;
      }
    }
    const size_t slot = (static_cast<int64_t>(rendered) + 60) % targets.size();
    if (headless) {
      if (rendered == 0) {
        for (auto &token : pending) {
          if (token) {
            if (!token.WaitHost())
              return 1;
            token = {};
          }
        }
        benchmark_start = std::chrono::steady_clock::now();
      }
      if (pending[slot]) {
        if (!pending[slot].WaitHost())
          return 1;
        pending[slot] = {};
      }
    }
    grassland::rhix::Image image;
    if (headless)
      image = targets[slot];
    else
      image = window.GetNextImage();
    if (!image) {
      if (headless)
        return 1;
      fps_counter.Reset();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    auto context = device.CreateCommandContext();
    if (!context)
      return 1;
    if (!context.CmdClearColor(image, {0.6f, 0.7f, 0.8f, 1.0f}))
      return 1;
    if (!headless) {
      if (!context.CmdPresent(window))
        return 1;
    }
    if (!capture_directory.empty()) {
      last_frame = context.CmdDownloadImage(image);
      if (!last_frame)
        return 1;
      last_frame_desc = image.GetDesc();
      last_frame_space = window ? window.GetColorSpace() : COLOR_SPACE_SRGB_NONLINEAR;
      last_frame_white_scale = window ? window.GetHDRReferenceWhiteScale() : 1.0f;
    }
    auto token = graphics.Submit(std::move(context));

    if (!token)
      return 1;
    if (headless)
      pending[slot] = std::move(token);
    ++rendered;
    fps_counter.TickFrame();
    const auto now = std::chrono::steady_clock::now();
    if (!headless && now - title_time >= std::chrono::milliseconds(250)) {
      if (!window.SetTitle(fmt::format("LongMarch RHIX - FPS: {:.1f} - P: toggle VSync", fps_counter.GetFPS())))
        return 1;
      title_time = now;
    }
  }
  if (headless) {
    for (const auto &token : pending) {
      if (token) {
        if (!token.WaitHost())
          return 1;
      }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - benchmark_start).count();
    std::cout << fmt::format("window: headless 1280x720, {} frames, {:.6f} s, {:.2f} FPS (3 in flight, 60 warmup)\n",
                             rendered, seconds, rendered / seconds);
  }
  if (!capture_directory.empty()) {
    if (!last_frame || !last_frame.WaitHost())
      return 1;
    std::vector<uint8_t> pixels(last_frame.GetSize());
    if (!last_frame.CopyTo(pixels.data(), pixels.size()) ||
        !SaveImage(capture_directory, "window", pixels, last_frame_desc, last_frame_space, last_frame_white_scale))
      return 1;
  }
  return 0;
}
