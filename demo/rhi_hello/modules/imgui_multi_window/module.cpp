#include "module.h"

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

#include "../../save_image.h"
#include "grassland/rhi/rhi.h"
#include "imgui.h"

using namespace grassland::rhi;

int ImGuiMultiWindowModule::OnInit(const ModuleOptions &options) {
  const bool headless = options.headless;
  capture_directory_ = options.capture_directory;
  const auto font_file = options.font_file;
  const auto font_size = options.font_size;
  using namespace grassland::rhi;
  device_ = GetDevice();
  if (headless) {
    std::cout << "imgui_multi_window: skipped (requires desktop windows)\n";
    {
      RequestClose();
      return 0;
    }
  }

  last_frame_descs_ = std::array<ImageDesc, 2>{};
  last_frame_spaces_ = std::array<ColorSpace, 2>{};
  last_frame_white_scales_ = std::array<float, 2>{};
  queue_ = device_->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue_)
    return 1;

  visible_ = std::array<bool, 2>{true, true};
  colors_ = std::array<ClearValue, 2>{{{0.6f, 0.7f, 0.8f, 1.0f}, {0.12f, 0.16f, 0.22f, 1.0f}}};

  const char *titles[] = {"ImGui multi - Controls - I: UI; Esc: close", "ImGui multi - Plot - I: UI; Esc: close"};
  for (size_t i = 0; i < windows_.size(); ++i) {
    windows_[i] = device_->CreateGLFWWindow({600, 480, titles[i]});
    if (!windows_[i])
      return 1;
    glfwSetWindowPos(windows_[i]->GLFWWindow(), 60 + int(i) * 620, 100);
    if (!windows_[i]->EnableImGui(true, font_file, font_size))
      return 1;
    windows_[i]->KeyEvent().RegisterCallback([this, i](int key, int, int action, int) {
      if (action != GLFW_PRESS)
        return;
      if (key == GLFW_KEY_I) {
        visible_[i] = !visible_[i];
        std::cout << "Window " << i + 1 << " UI: " << std::boolalpha << visible_[i] << '\n';
      }
      if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(windows_[i]->GLFWWindow(), GLFW_TRUE);
    });
  }
  clicks_ = 0;
  text_ = std::array<char, 128>{};
  speed_ = 1.0f;
  amplitude_ = 0.7f;
  phase_ = 0.0f;
  paused_ = false;

  initialized_ = true;
  return 0;
}

int ImGuiMultiWindowModule::OnUpdate() {
  // Collect all UI feedback before recording any window's scene commands.
  if (windows_[0] && !windows_[0]->ShouldClose() && !windows_[0]->UpdateImGui([&] {
        if (!visible_[0])
          return;
        ImGui::SetNextWindowPos({20, 20}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({550, 400}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Controls");
        ImGui::TextUnformatted("Window 1: independent controls and background");
        if (ImGui::Button("Click me"))
          ++clicks_;
        ImGui::Text("Local clicks: %d", clicks_);
        ImGui::InputText("Local text", text_.data(), text_.size());
        ImGui::ColorEdit3("Background", colors_[0].color.float32.data());
        ImGui::Separator();
        ImGui::TextWrapped(
            "I hides only this window's UI. Escape or the title-bar close button closes only this window.");
        ImGui::End();
      }))
    return 1;
  if (windows_[1] && !windows_[1]->ShouldClose() && !windows_[1]->UpdateImGui([&] {
        if (!visible_[1])
          return;
        if (!paused_)
          phase_ += ImGui::GetIO().DeltaTime * speed_;
        std::array<float, 128> samples;
        for (size_t i = 0; i < samples.size(); ++i)
          samples[i] = amplitude_ * std::sin(phase_ + float(i) * 0.1f);
        ImGui::SetNextWindowPos({20, 20}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({550, 400}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Live plot");
        ImGui::TextUnformatted("Window 2: animated signal, separate ImGui context");
        ImGui::SliderFloat("Speed", &speed_, 0.1f, 5.0f);
        ImGui::SliderFloat("Amplitude", &amplitude_, 0.0f, 1.0f);
        ImGui::Checkbox("Pause", &paused_);
        ImGui::PlotLines("Signal", samples.data(), int(samples.size()), 0, nullptr, -1, 1, {0, 160});
        ImGui::ColorEdit3("Background", colors_[1].color.float32.data());
        ImGui::TextUnformatted("I: toggle this UI; Escape: close this window");
        ImGui::End();
      }))
    return 1;

  return 0;
}

int ImGuiMultiWindowModule::OnRender() {
  bool open = false;
  bool submitted = false;
  for (size_t i = 0; i < windows_.size(); ++i) {
    auto &window = windows_[i];
    if (!window)
      continue;
    if (window->ShouldClose()) {
      if (pending_[i]) {
        if (!pending_[i]->WaitHost())
          return 1;
        pending_[i].Release();
      }
      window.Release();
      continue;
    }
    open = true;
    if (window->IsSwapChainInvalid()) {
      if (!window->RebuildSwapChain())
        return 1;
    }
    ObjPtr<IImage> image = window->GetNextImage();
    if (!image)
      continue;
    ObjPtr<ICommandContext> commands = device_->CreateCommandContext();
    if (!commands)
      return 1;
    if (!commands->CmdClearImage(image.Get(), colors_[i]))
      return 1;
    if (!commands->CmdPresent(window.Get()))
      return 1;
    if (!capture_directory_.empty()) {
      last_frames_[i] = commands->CmdDownloadImage(image.Get());
      if (!last_frames_[i])
        return 1;
      last_frame_descs_[i] = image->GetDesc();
      last_frame_spaces_[i] = window->GetColorSpace();
      last_frame_white_scales_[i] = window->GetHDRReferenceWhiteScale();
    }
    pending_[i] = queue_->Submit(commands.Get());

    if (!pending_[i])
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
  if (initialized_) {
    result = [&]() -> int {
      for (auto &token : pending_) {
        if (token) {
          if (!token->WaitHost())
            return 1;
        }
      }
      if (!capture_directory_.empty()) {
        for (size_t i = 0; i < last_frames_.size(); ++i) {
          auto &last_frame = last_frames_[i];
          if (!last_frame || !last_frame->WaitHost())
            return 1;
          std::vector<uint8_t> pixels(last_frame->GetSize());
          if (!last_frame->CopyTo(pixels.data(), pixels.size()) ||
              !SaveImage(capture_directory_, "imgui_multi_window_" + std::to_string(i), pixels, last_frame_descs_[i],
                         last_frame_spaces_[i], last_frame_white_scales_[i]))
            return 1;
        }
      }
      return 0;
    }();
  }
  paused_ = {};
  phase_ = {};
  amplitude_ = {};
  speed_ = {};
  text_ = {};
  clicks_ = {};
  pending_ = {};
  windows_ = {};
  colors_ = {};
  visible_ = {};
  queue_ = {};
  last_frame_white_scales_ = {};
  last_frame_spaces_ = {};
  last_frame_descs_ = {};
  last_frames_ = {};
  device_ = {};
  capture_directory_ = {};
  initialized_ = {};
  return result;
}
