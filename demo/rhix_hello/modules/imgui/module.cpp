#include "module.h"

#include <fmt/format.h>

#include <chrono>
#include <iostream>
#include <thread>
#include <utility>

#include "../../save_image.h"
#include "grassland/rhi/rhix/rhix.h"
#include "grassland/util/fps_counter.h"
#include "imgui.h"

using namespace grassland::rhi;
using namespace grassland::rhix;

int ImGuiModule::OnInit(const ModuleOptions &options) {
  const bool headless = options.headless;
  capture_directory_ = options.capture_directory;
  const auto font_file = options.font_file;
  const auto font_size = options.font_size;
  using namespace grassland::rhi;
  using namespace grassland::rhix;
  device_ = GetDevice();

  last_frame_desc_ = ImageDesc{};
  last_frame_space_ = COLOR_SPACE_SRGB_NONLINEAR;
  last_frame_white_scale_ = 1.0f;
  queue_ = device_.GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue_)
    return 1;
  requested_ = PRESENT_MODE_VSYNC;
  present_mode_changed_ = false;
  record_ui_ = true;
  window_ = device_.CreateGLFWWindow({1280, 720, "ImGui - I: toggle UI recording; P: VSync"});
  if (!window_)
    return 1;
  if (!window_.EnableImGui(true, font_file, font_size))
    return 1;
  window_.KeyEvent().RegisterCallback([this](int key, int, int action, int) {
    if (action != GLFW_PRESS)
      return;
    if (key == GLFW_KEY_ESCAPE)
      glfwSetWindowShouldClose(window_.GLFWWindow(), GLFW_TRUE);
    if (key == GLFW_KEY_P) {
      requested_ = requested_ == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
      present_mode_changed_ = true;
    }
    if (key == GLFW_KEY_I) {
      record_ui_ = !record_ui_;
      std::cout << "ImGui recording: " << std::boolalpha << record_ui_ << '\n';
    }
  });
  clicks_ = 0;
  show_demo_ = true;

  title_time_ = std::chrono::steady_clock::now();

  initialized_ = true;
  return 0;
}

int ImGuiModule::OnUpdate() {
  if (window_.ShouldClose()) {
    RequestClose();
    return 0;
  }
  if (!window_.UpdateImGui([&] {
        // Keep the context enabled: an empty frame must not replay previous UI.
        if (!record_ui_)
          return;
        ImGui::Begin("ImGui recording");
        ImGui::TextUnformatted("I: toggle UI recording; P: VSync; Escape: close");
        if (ImGui::Button("Click me"))
          ++clicks_;
        ImGui::Text("Clicks: %d", clicks_);
        ImGui::Checkbox("Show ImGui demo", &show_demo_);
        ImGui::End();
        if (show_demo_)
          ImGui::ShowDemoWindow(&show_demo_);
      }))
    return 1;
  if (window_.IsSwapChainInvalid()) {
    if (!window_.RebuildSwapChain())
      return 1;
  }
  if (present_mode_changed_) {
    present_mode_changed_ = false;
    if (!window_.SetPresentMode(requested_))
      return 1;
  }

  return 0;
}

int ImGuiModule::OnRender() {
  auto image = window_.GetNextImage();
  if (!image) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return 0;
  }
  auto commands = device_.CreateCommandContext();
  if (!commands)
    return 1;
  if (!commands.CmdClearImage(image, {0.6f, 0.7f, 0.8f, 1.0f}))
    return 1;
  if (!commands.CmdPresent(window_))
    return 1;
  if (!capture_directory_.empty()) {
    last_frame_ = commands.CmdDownloadImage(image);
    if (!last_frame_)
      return 1;
    last_frame_desc_ = image.GetDesc();
    last_frame_space_ = window_ ? window_.GetColorSpace() : COLOR_SPACE_SRGB_NONLINEAR;
    last_frame_white_scale_ = window_ ? window_.GetHDRReferenceWhiteScale() : 1.0f;
  }
  auto done = queue_.Submit(std::move(commands));

  if (!done)
    return 1;
  CompleteFrame();
  fps_.TickFrame();
  auto now = std::chrono::steady_clock::now();
  if (now - title_time_ >= std::chrono::milliseconds(250)) {
    if (!window_.SetTitle(fmt::format("ImGui - FPS: {:.1f} - UI recording: {} - I: toggle", fps_.GetFPS(), record_ui_)))
      return 1;
    title_time_ = now;
  }

  return 0;
}

int ImGuiModule::OnTerminate() {
  int result = 0;
  if (initialized_) {
    result = [&]() -> int {
      if (!capture_directory_.empty()) {
        if (!last_frame_ || !last_frame_.WaitHost())
          return 1;
        std::vector<uint8_t> pixels(last_frame_.GetSize());
        if (!last_frame_.CopyTo(pixels.data(), pixels.size()) ||
            !SaveImage(capture_directory_, "imgui", pixels, last_frame_desc_, last_frame_space_,
                       last_frame_white_scale_))
          return 1;
      }
      return 0;
    }();
  }
  title_time_ = {};
  fps_ = {};
  show_demo_ = {};
  clicks_ = {};
  window_ = {};
  record_ui_ = {};
  present_mode_changed_ = {};
  requested_ = {};
  queue_ = {};
  last_frame_white_scale_ = {};
  last_frame_space_ = {};
  last_frame_desc_ = {};
  last_frame_ = {};
  device_ = {};
  capture_directory_ = {};
  initialized_ = {};
  return result;
}
