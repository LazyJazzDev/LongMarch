#include "../../module.h"

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <utility>

#include "../../save_image.h"
#include "grassland/rhi/rhix/rhix.h"
#include "imgui.h"

int ImGuiMultiWindowModule::RunImpl(const ModuleOptions &options) {
  const int frames = options.frames;
  const bool headless = options.headless;
  const auto &capture_directory = options.capture_directory;
  const auto font_file = options.font_file;
  const auto font_size = options.font_size;
  using namespace grassland::rhi;
  using namespace grassland::rhix;
  const auto &device = GetDevice();
  if (headless) {
    std::cout << "imgui_multi_window: skipped (requires desktop windows)\n";
    return 0;
  }
  std::array<grassland::rhix::DownloadToken, 2> last_frames;
  std::array<ImageDesc, 2> last_frame_descs{};
  std::array<ColorSpace, 2> last_frame_spaces{};
  std::array<float, 2> last_frame_white_scales{};
  grassland::rhix::Queue queue = device.GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue)
    return 1;

  std::array<bool, 2> visible{true, true};
  std::array<ClearValue, 2> colors{{{0.6f, 0.7f, 0.8f, 1.0f}, {0.12f, 0.16f, 0.22f, 1.0f}}};
  std::array<grassland::rhix::Window, 2> windows;
  std::array<grassland::rhix::SubmissionToken, 2> pending;
  const char *titles[] = {"ImGui multi - Controls - I: UI; Esc: close", "ImGui multi - Plot - I: UI; Esc: close"};
  for (size_t i = 0; i < windows.size(); ++i) {
    windows[i] = device.CreateGLFWWindow({600, 480, titles[i]});
    if (!windows[i])
      return 1;
    glfwSetWindowPos(windows[i].GLFWWindow(), 60 + int(i) * 620, 100);
    if (!windows[i].EnableImGui(true, font_file, font_size))
      return 1;
    windows[i].KeyEvent().RegisterCallback([&, i](int key, int, int action, int) {
      if (action != GLFW_PRESS)
        return;
      if (key == GLFW_KEY_I) {
        visible[i] = !visible[i];
        std::cout << "Window " << i + 1 << " UI: " << std::boolalpha << visible[i] << '\n';
      }
      if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(windows[i].GLFWWindow(), GLFW_TRUE);
    });
  }
  int clicks = 0;
  std::array<char, 128> text{};
  float speed = 1.0f, amplitude = 0.7f, phase = 0.0f;
  bool paused = false;

  for (int rendered = 0; !frames || rendered < frames;) {
    if (!PollEvents())
      break;
    // Collect all UI feedback before recording any window's scene commands.
    if (windows[0] && !windows[0].ShouldClose() && !windows[0].UpdateImGui([&] {
          if (!visible[0])
            return;
          ImGui::SetNextWindowPos({20, 20}, ImGuiCond_FirstUseEver);
          ImGui::SetNextWindowSize({550, 400}, ImGuiCond_FirstUseEver);
          ImGui::Begin("Controls");
          ImGui::TextUnformatted("Window 1: independent controls and background");
          if (ImGui::Button("Click me"))
            ++clicks;
          ImGui::Text("Local clicks: %d", clicks);
          ImGui::InputText("Local text", text.data(), text.size());
          ImGui::ColorEdit3("Background", colors[0].color.float32.data());
          ImGui::Separator();
          ImGui::TextWrapped(
              "I hides only this window's UI. Escape or the title-bar close button closes only this window.");
          ImGui::End();
        }))
      return 1;
    if (windows[1] && !windows[1].ShouldClose() && !windows[1].UpdateImGui([&] {
          if (!visible[1])
            return;
          if (!paused)
            phase += ImGui::GetIO().DeltaTime * speed;
          std::array<float, 128> samples;
          for (size_t i = 0; i < samples.size(); ++i)
            samples[i] = amplitude * std::sin(phase + float(i) * 0.1f);
          ImGui::SetNextWindowPos({20, 20}, ImGuiCond_FirstUseEver);
          ImGui::SetNextWindowSize({550, 400}, ImGuiCond_FirstUseEver);
          ImGui::Begin("Live plot");
          ImGui::TextUnformatted("Window 2: animated signal, separate ImGui context");
          ImGui::SliderFloat("Speed", &speed, 0.1f, 5.0f);
          ImGui::SliderFloat("Amplitude", &amplitude, 0.0f, 1.0f);
          ImGui::Checkbox("Pause", &paused);
          ImGui::PlotLines("Signal", samples.data(), int(samples.size()), 0, nullptr, -1, 1, {0, 160});
          ImGui::ColorEdit3("Background", colors[1].color.float32.data());
          ImGui::TextUnformatted("I: toggle this UI; Escape: close this window");
          ImGui::End();
        }))
      return 1;
    bool open = false;
    bool submitted = false;
    for (size_t i = 0; i < windows.size(); ++i) {
      auto &window = windows[i];
      if (!window)
        continue;
      if (window.ShouldClose()) {
        if (pending[i]) {
          if (!pending[i].WaitHost())
            return 1;
          pending[i].Release();
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
      grassland::rhix::CommandContext commands = device.CreateCommandContext();
      if (!commands)
        return 1;
      if (!commands.CmdClearImage(image, colors[i]))
        return 1;
      if (!commands.CmdPresent(window))
        return 1;
      if (!capture_directory.empty()) {
        last_frames[i] = commands.CmdDownloadImage(image);
        if (!last_frames[i])
          return 1;
        last_frame_descs[i] = image.GetDesc();
        last_frame_spaces[i] = window.GetColorSpace();
        last_frame_white_scales[i] = window.GetHDRReferenceWhiteScale();
      }
      pending[i] = queue.Submit(std::move(commands));

      if (!pending[i])
        return 1;
      submitted = true;
    }
    if (!open)
      break;
    if (submitted)
      ++rendered;
    else
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  for (auto &token : pending) {
    if (token) {
      if (!token.WaitHost())
        return 1;
    }
  }
  if (!capture_directory.empty()) {
    for (size_t i = 0; i < last_frames.size(); ++i) {
      auto &last_frame = last_frames[i];
      if (!last_frame || !last_frame.WaitHost())
        return 1;
      std::vector<uint8_t> pixels(last_frame.GetSize());
      if (!last_frame.CopyTo(pixels.data(), pixels.size()) ||
          !SaveImage(capture_directory, "imgui_multi_window_" + std::to_string(i), pixels, last_frame_descs[i],
                     last_frame_spaces[i], last_frame_white_scales[i]))
        return 1;
    }
  }
  return 0;
}
