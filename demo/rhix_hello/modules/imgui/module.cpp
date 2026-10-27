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

struct ImGuiModule::State {
  bool initialized{};
  std::string capture_directory{};
  grassland::rhix::Device device{};
  grassland::rhix::DownloadToken last_frame{};
  ImageDesc last_frame_desc{};
  ColorSpace last_frame_space{};
  float last_frame_white_scale{};
  Queue queue{};
  PresentMode requested{};
  bool present_mode_changed{};
  bool record_ui{};
  Window window{};
  int clicks{};
  bool show_demo{};
  grassland::FPSCounter fps{};
  std::chrono::steady_clock::time_point title_time{};
};

ImGuiModule::ImGuiModule(grassland::rhix::Core core, grassland::rhix::Device device)
    : ModuleBase(std::move(core), std::move(device)) {
}

ImGuiModule::~ImGuiModule() = default;

int ImGuiModule::OnInit(const ModuleOptions &options) {
  state_ = std::make_unique<State>();

  const bool headless = options.headless;
  state_->capture_directory = options.capture_directory;
  const auto font_file = options.font_file;
  const auto font_size = options.font_size;
  using namespace grassland::rhi;
  using namespace grassland::rhix;
  state_->device = GetDevice();

  state_->last_frame_desc = ImageDesc{};
  state_->last_frame_space = COLOR_SPACE_SRGB_NONLINEAR;
  state_->last_frame_white_scale = 1.0f;
  state_->queue = state_->device.GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!state_->queue)
    return 1;
  state_->requested = PRESENT_MODE_VSYNC;
  state_->present_mode_changed = false;
  state_->record_ui = true;
  state_->window = state_->device.CreateGLFWWindow({1280, 720, "ImGui - I: toggle UI recording; P: VSync"});
  if (!state_->window)
    return 1;
  if (!state_->window.EnableImGui(true, font_file, font_size))
    return 1;
  state_->window.KeyEvent().RegisterCallback([this](int key, int, int action, int) {
    if (action != GLFW_PRESS)
      return;
    if (key == GLFW_KEY_ESCAPE)
      glfwSetWindowShouldClose(state_->window.GLFWWindow(), GLFW_TRUE);
    if (key == GLFW_KEY_P) {
      state_->requested = state_->requested == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
      state_->present_mode_changed = true;
    }
    if (key == GLFW_KEY_I) {
      state_->record_ui = !state_->record_ui;
      std::cout << "ImGui recording: " << std::boolalpha << state_->record_ui << '\n';
    }
  });
  state_->clicks = 0;
  state_->show_demo = true;

  state_->title_time = std::chrono::steady_clock::now();

  state_->initialized = true;
  return 0;
}

int ImGuiModule::OnUpdate() {
  if (state_->window.ShouldClose()) {
    RequestClose();
    return 0;
  }
  if (!state_->window.UpdateImGui([&] {
        // Keep the context enabled: an empty frame must not replay previous UI.
        if (!state_->record_ui)
          return;
        ImGui::Begin("ImGui recording");
        ImGui::TextUnformatted("I: toggle UI recording; P: VSync; Escape: close");
        if (ImGui::Button("Click me"))
          ++state_->clicks;
        ImGui::Text("Clicks: %d", state_->clicks);
        ImGui::Checkbox("Show ImGui demo", &state_->show_demo);
        ImGui::End();
        if (state_->show_demo)
          ImGui::ShowDemoWindow(&state_->show_demo);
      }))
    return 1;
  if (state_->window.IsSwapChainInvalid()) {
    if (!state_->window.RebuildSwapChain())
      return 1;
  }
  if (state_->present_mode_changed) {
    state_->present_mode_changed = false;
    if (!state_->window.SetPresentMode(state_->requested))
      return 1;
  }

  return 0;
}

int ImGuiModule::OnRender() {
  auto image = state_->window.GetNextImage();
  if (!image) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return 0;
  }
  auto commands = state_->device.CreateCommandContext();
  if (!commands)
    return 1;
  if (!commands.CmdClearImage(image, {0.6f, 0.7f, 0.8f, 1.0f}))
    return 1;
  if (!commands.CmdPresent(state_->window))
    return 1;
  if (!state_->capture_directory.empty()) {
    state_->last_frame = commands.CmdDownloadImage(image);
    if (!state_->last_frame)
      return 1;
    state_->last_frame_desc = image.GetDesc();
    state_->last_frame_space = state_->window ? state_->window.GetColorSpace() : COLOR_SPACE_SRGB_NONLINEAR;
    state_->last_frame_white_scale = state_->window ? state_->window.GetHDRReferenceWhiteScale() : 1.0f;
  }
  auto done = state_->queue.Submit(std::move(commands));

  if (!done)
    return 1;
  CompleteFrame();
  state_->fps.TickFrame();
  auto now = std::chrono::steady_clock::now();
  if (now - state_->title_time >= std::chrono::milliseconds(250)) {
    if (!state_->window.SetTitle(
            fmt::format("ImGui - FPS: {:.1f} - UI recording: {} - I: toggle", state_->fps.GetFPS(), state_->record_ui)))
      return 1;
    state_->title_time = now;
  }

  return 0;
}

int ImGuiModule::OnTerminate() {
  int result = 0;
  if (state_ && state_->initialized) {
    result = [&]() -> int {
      if (!state_->capture_directory.empty()) {
        if (!state_->last_frame || !state_->last_frame.WaitHost())
          return 1;
        std::vector<uint8_t> pixels(state_->last_frame.GetSize());
        if (!state_->last_frame.CopyTo(pixels.data(), pixels.size()) ||
            !SaveImage(state_->capture_directory, "imgui", pixels, state_->last_frame_desc, state_->last_frame_space,
                       state_->last_frame_white_scale))
          return 1;
      }
      return 0;
    }();
  }
  state_.reset();
  return result;
}
