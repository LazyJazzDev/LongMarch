#include "module.h"

#include <fmt/format.h>

#include <array>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "../../save_image.h"
#include "grassland/rhi/rhi.h"
#include "grassland/util/fps_counter.h"

using namespace grassland::rhi;

struct WindowModule::State {
  bool initialized{};
  bool headless{};
  std::string capture_directory{};
  grassland::rhi::IDevice *device{};
  PresentMode requested{};
  bool present_mode_changed{};
  ObjPtr<IWindow> window{};
  ObjPtr<IQueue> graphics{};
  ObjPtr<IDownloadToken> last_frame{};
  ImageDesc last_frame_desc{};
  ColorSpace last_frame_space{};
  float last_frame_white_scale{};
  std::array<ObjPtr<IImage>, 3> targets{};
  std::array<ObjPtr<ISubmissionToken>, 3> pending{};
  grassland::FPSCounter fps_counter{};
  std::chrono::steady_clock::time_point title_time{};
  std::chrono::steady_clock::time_point benchmark_start{};
};

WindowModule::WindowModule(grassland::rhi::ObjPtr<grassland::rhi::ICore> core,
                           grassland::rhi::ObjPtr<grassland::rhi::IDevice> device)
    : ModuleBase(std::move(core), std::move(device)) {
}

WindowModule::~WindowModule() = default;

int WindowModule::OnInit(const ModuleOptions &options) {
  state_ = std::make_unique<State>();

  state_->headless = options.headless;
  state_->capture_directory = options.capture_directory;
  const auto api = GetBackendAPI();
  using namespace grassland::rhi;
  state_->device = GetDevice();
  auto *core = GetCore();
  const auto devices = core->GetPhysicalDevices();
  std::cout << "Backend: " << BackendAPIName(api) << '\n' << std::boolalpha;
  for (size_t id = 0; id < devices.size(); ++id) {
    const auto &info = devices[id];
    std::cout << '[' << id + 1 << "] " << info.name << " (" << PhysicalDeviceTypeName(info.type) << ")\n"
              << "  Dedicated: " << info.dedicated_memory / (1024 * 1024)
              << " MiB, shared/unified: " << info.unified_memory / (1024 * 1024) << " MiB\n"
              << "  Ray tracing pipeline: " << info.ray_tracing_pipeline_feature
              << ", ray query: " << info.ray_query_feature << ", geometry shader: " << info.geometry_shader_feature
              << '\n';
  }

  std::cout << "Selected: " << state_->device->Name() << '\n'
            << "Enabled: ray tracing pipeline=" << state_->device->SupportRayTracingPipeline()
            << ", ray query=" << state_->device->SupportRayQuery()
            << ", geometry shader=" << state_->device->SupportGeometryShader() << '\n';
  state_->requested = PRESENT_MODE_VSYNC;
  state_->present_mode_changed = false;

  if (!state_->headless) {
    state_->window = state_->device->CreateGLFWWindow({1280, 720, "LongMarch RHI - P: toggle VSync"});
    if (!state_->window)
      return 1;
  }

  state_->graphics = state_->device->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!state_->graphics)
    return 1;
  if (!state_->headless) {
    state_->window->KeyEvent().RegisterCallback([this](int key, int, int action, int) {
      if (action != GLFW_PRESS)
        return;
      if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(state_->window->GLFWWindow(), GLFW_TRUE);
      if (key == GLFW_KEY_P) {
        state_->requested = state_->requested == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
        state_->present_mode_changed = true;
      }
    });
  }
  using namespace grassland::rhi;

  state_->last_frame_desc = ImageDesc{};
  state_->last_frame_space = COLOR_SPACE_SRGB_NONLINEAR;
  state_->last_frame_white_scale = 1.0f;
  if (!state_->headless) {
    const auto options = state_->window->GetSupportedPresentationOptions();
    if (options.empty())
      return 1;
    std::cout << "Supported presentation combinations (SDR):\n";
    for (const auto &option : options) {
      std::cout << "  " << PresentModeName(option.present_mode) << " + " << ImageFormatName(option.format) << '\n';
    }
  }

  if (state_->headless) {
    for (auto &target : state_->targets) {
      target = state_->device->CreateImage({1280, 720, IMAGE_FORMAT_R8G8B8A8_UNORM});
      if (!target)
        return 1;
    }
  }

  state_->title_time = std::chrono::steady_clock::now();
  // Warm up allocations and pipelines before measuring completed GPU work.
  state_->benchmark_start = std::chrono::steady_clock::now();
  SetWarmupFrames(state_->headless ? 60 : 0);

  state_->initialized = true;
  return 0;
}

int WindowModule::OnUpdate() {
  if (!state_->headless) {
    if (state_->window->ShouldClose()) {
      RequestClose();
      return 0;
    }
    // Previous loop locals are gone; rebuild only before recording this frame.
    if (state_->window->IsSwapChainInvalid()) {
      if (!state_->window->RebuildSwapChain())
        return 1;
    }
    if (state_->present_mode_changed) {
      state_->present_mode_changed = false;
      if (!state_->window->SetPresentMode(state_->requested))
        return 1;
      std::cout << "Present mode: " << PresentModeName(state_->window->GetPresentMode()) << std::endl;
    }
  }

  return 0;
}

int WindowModule::OnRender() {
  const size_t slot = (static_cast<int64_t>(GetFrameIndex()) + 60) % state_->targets.size();
  if (state_->headless) {
    if (GetFrameIndex() == 0) {
      for (auto &token : state_->pending) {
        if (token) {
          if (!token->WaitHost())
            return 1;
          token = {};
        }
      }
      state_->benchmark_start = std::chrono::steady_clock::now();
    }
    if (state_->pending[slot]) {
      if (!state_->pending[slot]->WaitHost())
        return 1;
      state_->pending[slot] = {};
    }
  }
  ObjPtr<IImage> image;
  if (state_->headless)
    image = state_->targets[slot];
  else
    image = state_->window->GetNextImage();
  if (!image) {
    if (state_->headless)
      return 1;
    state_->fps_counter.Reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return 0;
  }
  ObjPtr<ICommandContext> context = state_->device->CreateCommandContext();
  if (!context)
    return 1;
  if (!context->CmdClearImage(image.Get(), {0.6f, 0.7f, 0.8f, 1.0f}))
    return 1;
  if (!state_->headless) {
    if (!context->CmdPresent(state_->window.Get()))
      return 1;
  }
  if (!state_->capture_directory.empty()) {
    state_->last_frame = context->CmdDownloadImage(image.Get());
    if (!state_->last_frame)
      return 1;
    state_->last_frame_desc = image->GetDesc();
    state_->last_frame_space = state_->window ? state_->window->GetColorSpace() : COLOR_SPACE_SRGB_NONLINEAR;
    state_->last_frame_white_scale = state_->window ? state_->window->GetHDRReferenceWhiteScale() : 1.0f;
  }
  // Submit ends recording and borrows the caller's context reference.
  ObjPtr<ISubmissionToken> token = state_->graphics->Submit(context.Get());

  if (!token)
    return 1;
  // Local containers release caller references; accepted work retains its resources.
  if (state_->headless)
    state_->pending[slot] = std::move(token);
  CompleteFrame();
  state_->fps_counter.TickFrame();
  const auto now = std::chrono::steady_clock::now();
  if (!state_->headless && now - state_->title_time >= std::chrono::milliseconds(250)) {
    if (!state_->window->SetTitle(
            fmt::format("LongMarch RHI - FPS: {:.1f} - P: toggle VSync", state_->fps_counter.GetFPS())))
      return 1;
    state_->title_time = now;
  }

  return 0;
}

int WindowModule::OnTerminate() {
  int result = 0;
  if (state_ && state_->initialized) {
    result = [&]() -> int {
      if (state_->headless) {
        for (const auto &token : state_->pending) {
          if (token) {
            if (!token->WaitHost())
              return 1;
          }
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - state_->benchmark_start).count();
        std::cout << fmt::format(
            "window: headless 1280x720, {} frames, {:.6f} s, {:.2f} FPS (3 in flight, 60 warmup)\n", GetFrameIndex(),
            seconds, GetFrameIndex() / seconds);
      }
      if (!state_->capture_directory.empty()) {
        if (!state_->last_frame || !state_->last_frame->WaitHost())
          return 1;
        std::vector<uint8_t> pixels(state_->last_frame->GetSize());
        if (!state_->last_frame->CopyTo(pixels.data(), pixels.size()) ||
            !SaveImage(state_->capture_directory, "window", pixels, state_->last_frame_desc, state_->last_frame_space,
                       state_->last_frame_white_scale))
          return 1;
      }
      return 0;
    }();
  }
  state_.reset();
  return result;
}
