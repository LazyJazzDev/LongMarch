#include "../../module.h"

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

#include "grassland/rhi/rhi.h"
#include "imgui.h"

int RunImGuiMultiWindow(grassland::rhi::BackendAPI api, int frames, bool headless) {
  using namespace grassland::rhi;
  if (headless) {
    std::cout << "imgui_multi_window: skipped (requires desktop windows)\n";
    return 0;
  }
  ObjPtr<ICore> core = CreateCore(api);
  if (!core)
    return 1;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  if (!device)
    return 1;
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue)
    return 1;

  std::array<bool, 2> visible{true, true};
  std::array<std::array<float, 4>, 2> colors{{{0.6f, 0.7f, 0.8f, 1}, {0.12f, 0.16f, 0.22f, 1}}};
  std::array<ObjPtr<IWindow>, 2> windows;
  std::array<ObjPtr<ISubmissionToken>, 2> pending;
  const char *titles[] = {"ImGui multi - Controls - I: UI; Esc: close", "ImGui multi - Plot - I: UI; Esc: close"};
  for (size_t i = 0; i < windows.size(); ++i) {
    windows[i] = device->CreateGLFWWindow({600, 480, titles[i]});
    if (!windows[i])
      return 1;
    glfwSetWindowPos(windows[i]->GLFWWindow(), 60 + int(i) * 620, 100);
    if (!windows[i]->EnableImGui())
      return 1;
    windows[i]->KeyEvent().RegisterCallback([&, i](int key, int, int action, int) {
      if (action != GLFW_PRESS)
        return;
      if (key == GLFW_KEY_I) {
        visible[i] = !visible[i];
        std::cout << "Window " << i + 1 << " UI: " << std::boolalpha << visible[i] << '\n';
      }
      if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(windows[i]->GLFWWindow(), GLFW_TRUE);
    });
  }
  windows[0]->ImGuiEvent().RegisterCallback([&, clicks = 0, text = std::array<char, 128>{}]() mutable {
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
    ImGui::ColorEdit3("Background", colors[0].data());
    ImGui::Separator();
    ImGui::TextWrapped("I hides only this window's UI. Escape or the title-bar close button closes only this window.");
    ImGui::End();
  });
  windows[1]->ImGuiEvent().RegisterCallback(
      [&, speed = 1.0f, amplitude = 0.7f, paused = false, phase = 0.0f]() mutable {
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
        ImGui::ColorEdit3("Background", colors[1].data());
        ImGui::TextUnformatted("I: toggle this UI; Escape: close this window");
        ImGui::End();
      });

  for (int rendered = 0; !frames || rendered < frames;) {
    if (!PollEvents())
      return 1;
    bool open = false;
    bool submitted = false;
    for (size_t i = 0; i < windows.size(); ++i) {
      auto &window = windows[i];
      if (!window)
        continue;
      if (window->ShouldClose()) {
        if (pending[i]) {
          if (!pending[i]->WaitHost())
            return 1;
          pending[i].Release();
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
      ObjPtr<ICommandContext> commands = device->CreateCommandContext();
      if (!commands)
        return 1;
      if (!commands->CmdClearColor(image.Get(), colors[i]))
        return 1;
      if (!commands->CmdPresent(window.Get()))
        return 1;
      pending[i] = queue->Submit(commands.Get());
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
      if (!token->WaitHost())
        return 1;
    }
  }
  return 0;
}
