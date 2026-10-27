#include "module.h"

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <utility>

#include "../../save_image.h"
#include "grassland/rhi/rhix/rhix.h"
#include "imgui.h"

using namespace grassland::rhi;
using namespace grassland::rhix;

struct ImGuiMultiWindowModule::State {
  bool initialized{};
  std::string capture_directory{};
  grassland::rhix::Device device{};
  std::array<grassland::rhix::DownloadToken, 2> last_frames{};
  std::array<ImageDesc, 2> last_frame_descs{};
  std::array<ColorSpace, 2> last_frame_spaces{};
  std::array<float, 2> last_frame_white_scales{};
  grassland::rhix::Queue queue{};
  std::array<bool, 2> visible{};
  std::array<ClearValue, 2> colors{};
  std::array<grassland::rhix::Window, 2> windows{};
  std::array<grassland::rhix::SubmissionToken, 2> pending{};
  int clicks{};
  std::array<char, 128> text{};
  float speed{};
  float amplitude{};
  float phase{};
  bool paused{};
};

ImGuiMultiWindowModule::ImGuiMultiWindowModule(grassland::rhix::Core core, grassland::rhix::Device device)
    : ModuleBase(std::move(core), std::move(device)) {
}

ImGuiMultiWindowModule::~ImGuiMultiWindowModule() = default;

int ImGuiMultiWindowModule::OnInit(const ModuleOptions &options) {
  state_ = std::make_unique<State>();

  const bool headless = options.headless;
  state_->capture_directory = options.capture_directory;
  const auto font_file = options.font_file;
  const auto font_size = options.font_size;
  using namespace grassland::rhi;
  using namespace grassland::rhix;
  state_->device = GetDevice();
  if (headless) {
    std::cout << "imgui_multi_window: skipped (requires desktop windows)\n";
    {
      RequestClose();
      return 0;
    }
  }

  state_->last_frame_descs = std::array<ImageDesc, 2>{};
  state_->last_frame_spaces = std::array<ColorSpace, 2>{};
  state_->last_frame_white_scales = std::array<float, 2>{};
  state_->queue = state_->device.GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!state_->queue)
    return 1;

  state_->visible = std::array<bool, 2>{true, true};
  state_->colors = std::array<ClearValue, 2>{{{0.6f, 0.7f, 0.8f, 1.0f}, {0.12f, 0.16f, 0.22f, 1.0f}}};

  const char *titles[] = {"ImGui multi - Controls - I: UI; Esc: close", "ImGui multi - Plot - I: UI; Esc: close"};
  for (size_t i = 0; i < state_->windows.size(); ++i) {
    state_->windows[i] = state_->device.CreateGLFWWindow({600, 480, titles[i]});
    if (!state_->windows[i])
      return 1;
    glfwSetWindowPos(state_->windows[i].GLFWWindow(), 60 + int(i) * 620, 100);
    if (!state_->windows[i].EnableImGui(true, font_file, font_size))
      return 1;
    state_->windows[i].KeyEvent().RegisterCallback([this, i](int key, int, int action, int) {
      if (action != GLFW_PRESS)
        return;
      if (key == GLFW_KEY_I) {
        state_->visible[i] = !state_->visible[i];
        std::cout << "Window " << i + 1 << " UI: " << std::boolalpha << state_->visible[i] << '\n';
      }
      if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(state_->windows[i].GLFWWindow(), GLFW_TRUE);
    });
  }
  state_->clicks = 0;
  state_->text = std::array<char, 128>{};
  state_->speed = 1.0f;
  state_->amplitude = 0.7f;
  state_->phase = 0.0f;
  state_->paused = false;

  state_->initialized = true;
  return 0;
}

int ImGuiMultiWindowModule::OnUpdate() {
  // Collect all UI feedback before recording any window's scene commands.
  if (state_->windows[0] && !state_->windows[0].ShouldClose() && !state_->windows[0].UpdateImGui([&] {
        if (!state_->visible[0])
          return;
        ImGui::SetNextWindowPos({20, 20}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({550, 400}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Controls");
        ImGui::TextUnformatted("Window 1: independent controls and background");
        if (ImGui::Button("Click me"))
          ++state_->clicks;
        ImGui::Text("Local clicks: %d", state_->clicks);
        ImGui::InputText("Local text", state_->text.data(), state_->text.size());
        ImGui::ColorEdit3("Background", state_->colors[0].color.float32.data());
        ImGui::Separator();
        ImGui::TextWrapped(
            "I hides only this window's UI. Escape or the title-bar close button closes only this window.");
        ImGui::End();
      }))
    return 1;
  if (state_->windows[1] && !state_->windows[1].ShouldClose() && !state_->windows[1].UpdateImGui([&] {
        if (!state_->visible[1])
          return;
        if (!state_->paused)
          state_->phase += ImGui::GetIO().DeltaTime * state_->speed;
        std::array<float, 128> samples;
        for (size_t i = 0; i < samples.size(); ++i)
          samples[i] = state_->amplitude * std::sin(state_->phase + float(i) * 0.1f);
        ImGui::SetNextWindowPos({20, 20}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({550, 400}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Live plot");
        ImGui::TextUnformatted("Window 2: animated signal, separate ImGui context");
        ImGui::SliderFloat("Speed", &state_->speed, 0.1f, 5.0f);
        ImGui::SliderFloat("Amplitude", &state_->amplitude, 0.0f, 1.0f);
        ImGui::Checkbox("Pause", &state_->paused);
        ImGui::PlotLines("Signal", samples.data(), int(samples.size()), 0, nullptr, -1, 1, {0, 160});
        ImGui::ColorEdit3("Background", state_->colors[1].color.float32.data());
        ImGui::TextUnformatted("I: toggle this UI; Escape: close this window");
        ImGui::End();
      }))
    return 1;

  return 0;
}

int ImGuiMultiWindowModule::OnRender() {
  bool open = false;
  bool submitted = false;
  for (size_t i = 0; i < state_->windows.size(); ++i) {
    auto &window = state_->windows[i];
    if (!window)
      continue;
    if (window.ShouldClose()) {
      if (state_->pending[i]) {
        if (!state_->pending[i].WaitHost())
          return 1;
        state_->pending[i].Release();
      }
      window.Release();
      continue;
    }
    open = true;
    if (window.IsSwapChainInvalid()) {
      if (!window.RebuildSwapChain())
        return 1;
    }
    grassland::rhix::Image image = window.GetNextImage();
    if (!image)
      continue;
    grassland::rhix::CommandContext commands = state_->device.CreateCommandContext();
    if (!commands)
      return 1;
    if (!commands.CmdClearImage(image, state_->colors[i]))
      return 1;
    if (!commands.CmdPresent(window))
      return 1;
    if (!state_->capture_directory.empty()) {
      state_->last_frames[i] = commands.CmdDownloadImage(image);
      if (!state_->last_frames[i])
        return 1;
      state_->last_frame_descs[i] = image.GetDesc();
      state_->last_frame_spaces[i] = window.GetColorSpace();
      state_->last_frame_white_scales[i] = window.GetHDRReferenceWhiteScale();
    }
    state_->pending[i] = state_->queue.Submit(std::move(commands));

    if (!state_->pending[i])
      return 1;
    submitted = true;
  }
  if (!open) {
    RequestClose();
    return 0;
  }
  if (submitted)
    CompleteFrame();
  else
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  return 0;
}

int ImGuiMultiWindowModule::OnTerminate() {
  int result = 0;
  if (state_ && state_->initialized) {
    result = [&]() -> int {
      for (auto &token : state_->pending) {
        if (token) {
          if (!token.WaitHost())
            return 1;
        }
      }
      if (!state_->capture_directory.empty()) {
        for (size_t i = 0; i < state_->last_frames.size(); ++i) {
          auto &last_frame = state_->last_frames[i];
          if (!last_frame || !last_frame.WaitHost())
            return 1;
          std::vector<uint8_t> pixels(last_frame.GetSize());
          if (!last_frame.CopyTo(pixels.data(), pixels.size()) ||
              !SaveImage(state_->capture_directory, "imgui_multi_window_" + std::to_string(i), pixels,
                         state_->last_frame_descs[i], state_->last_frame_spaces[i], state_->last_frame_white_scales[i]))
            return 1;
        }
      }
      return 0;
    }();
  }
  state_.reset();
  return result;
}
