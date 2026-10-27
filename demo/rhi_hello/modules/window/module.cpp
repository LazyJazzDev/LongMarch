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

int WindowModule::OnInit(const ModuleOptions &options) {
  headless_ = options.headless;
  capture_directory_ = options.capture_directory;
  const auto api = GetBackendAPI();
  using namespace grassland::rhi;
  device_ = GetDevice();
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

  std::cout << "Selected: " << device_->Name() << '\n'
            << "Enabled: ray tracing pipeline=" << device_->SupportRayTracingPipeline()
            << ", ray query=" << device_->SupportRayQuery() << ", geometry shader=" << device_->SupportGeometryShader()
            << '\n';
  requested_ = PRESENT_MODE_VSYNC;
  present_mode_changed_ = false;

  if (!headless_) {
    window_ = device_->CreateGLFWWindow({1280, 720, "LongMarch RHI - P: toggle VSync"});
    if (!window_)
      return 1;
  }

  graphics_ = device_->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!graphics_)
    return 1;
  if (!headless_) {
    window_->KeyEvent().RegisterCallback([this](int key, int, int action, int) {
      if (action != GLFW_PRESS)
        return;
      if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window_->GLFWWindow(), GLFW_TRUE);
      if (key == GLFW_KEY_P) {
        requested_ = requested_ == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
        present_mode_changed_ = true;
      }
    });
  }
  using namespace grassland::rhi;

  last_frame_desc_ = ImageDesc{};
  last_frame_space_ = COLOR_SPACE_SRGB_NONLINEAR;
  last_frame_white_scale_ = 1.0f;
  if (!headless_) {
    const auto options = window_->GetSupportedPresentationOptions();
    if (options.empty())
      return 1;
    std::cout << "Supported presentation combinations (SDR):\n";
    for (const auto &option : options) {
      std::cout << "  " << PresentModeName(option.present_mode) << " + " << ImageFormatName(option.format) << '\n';
    }
  }

  if (headless_) {
    for (auto &target : targets_) {
      target = device_->CreateImage({1280, 720, IMAGE_FORMAT_R8G8B8A8_UNORM});
      if (!target)
        return 1;
    }
  }

  title_time_ = std::chrono::steady_clock::now();
  // Warm up allocations and pipelines before measuring completed GPU work.
  benchmark_start_ = std::chrono::steady_clock::now();
  SetWarmupFrames(headless_ ? 60 : 0);

  initialized_ = true;
  return 0;
}

int WindowModule::OnUpdate() {
  if (!headless_) {
    if (window_->ShouldClose()) {
      RequestClose();
      return 0;
    }
    // Previous loop locals are gone; rebuild only before recording this frame.
    if (window_->IsSwapChainInvalid()) {
      if (!window_->RebuildSwapChain())
        return 1;
    }
    if (present_mode_changed_) {
      present_mode_changed_ = false;
      if (!window_->SetPresentMode(requested_))
        return 1;
      std::cout << "Present mode: " << PresentModeName(window_->GetPresentMode()) << std::endl;
    }
  }

  return 0;
}

int WindowModule::OnRender() {
  const size_t slot = (static_cast<int64_t>(GetFrameIndex()) + 60) % targets_.size();
  if (headless_) {
    if (GetFrameIndex() == 0) {
      for (auto &token : pending_) {
        if (token) {
          if (!token->WaitHost())
            return 1;
          token = {};
        }
      }
      benchmark_start_ = std::chrono::steady_clock::now();
    }
    if (pending_[slot]) {
      if (!pending_[slot]->WaitHost())
        return 1;
      pending_[slot] = {};
    }
  }
  ObjPtr<IImage> image;
  if (headless_)
    image = targets_[slot];
  else
    image = window_->GetNextImage();
  if (!image) {
    if (headless_)
      return 1;
    fps_counter_.Reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return 0;
  }
  ObjPtr<ICommandContext> context = device_->CreateCommandContext();
  if (!context)
    return 1;
  if (!context->CmdClearImage(image.Get(), {0.6f, 0.7f, 0.8f, 1.0f}))
    return 1;
  if (!headless_) {
    if (!context->CmdPresent(window_.Get()))
      return 1;
  }
  if (!capture_directory_.empty()) {
    last_frame_ = context->CmdDownloadImage(image.Get());
    if (!last_frame_)
      return 1;
    last_frame_desc_ = image->GetDesc();
    last_frame_space_ = window_ ? window_->GetColorSpace() : COLOR_SPACE_SRGB_NONLINEAR;
    last_frame_white_scale_ = window_ ? window_->GetHDRReferenceWhiteScale() : 1.0f;
  }
  // Submit ends recording and borrows the caller's context reference.
  ObjPtr<ISubmissionToken> token = graphics_->Submit(context.Get());

  if (!token)
    return 1;
  // Local containers release caller references; accepted work retains its resources.
  if (headless_)
    pending_[slot] = std::move(token);
  CompleteFrame();
  fps_counter_.TickFrame();
  const auto now = std::chrono::steady_clock::now();
  if (!headless_ && now - title_time_ >= std::chrono::milliseconds(250)) {
    if (!window_->SetTitle(fmt::format("LongMarch RHI - FPS: {:.1f} - P: toggle VSync", fps_counter_.GetFPS())))
      return 1;
    title_time_ = now;
  }

  return 0;
}

int WindowModule::OnTerminate() {
  int result = 0;
  if (initialized_) {
    result = [&]() -> int {
      if (headless_) {
        for (const auto &token : pending_) {
          if (token) {
            if (!token->WaitHost())
              return 1;
          }
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - benchmark_start_).count();
        std::cout << fmt::format(
            "window: headless 1280x720, {} frames, {:.6f} s, {:.2f} FPS (3 in flight, 60 warmup)\n", GetFrameIndex(),
            seconds, GetFrameIndex() / seconds);
      }
      if (!capture_directory_.empty()) {
        if (!last_frame_ || !last_frame_->WaitHost())
          return 1;
        std::vector<uint8_t> pixels(last_frame_->GetSize());
        if (!last_frame_->CopyTo(pixels.data(), pixels.size()) ||
            !SaveImage(capture_directory_, "window", pixels, last_frame_desc_, last_frame_space_,
                       last_frame_white_scale_))
          return 1;
      }
      return 0;
    }();
  }
  benchmark_start_ = {};
  title_time_ = {};
  fps_counter_ = {};
  pending_ = {};
  targets_ = {};
  last_frame_white_scale_ = {};
  last_frame_space_ = {};
  last_frame_desc_ = {};
  last_frame_ = {};
  graphics_ = {};
  window_ = {};
  present_mode_changed_ = {};
  requested_ = {};
  device_ = {};
  capture_directory_ = {};
  headless_ = {};
  initialized_ = {};
  return result;
}
