#include "../../module.h"

#include <fmt/format.h>

#include <chrono>
#include <iostream>
#include <thread>
#include <utility>

#include "grassland/rhi/rhix/rhix.h"
#include "grassland/util/fps_counter.h"
#include "imgui.h"

int RunImGui(grassland::rhi::BackendAPI api, int frames, bool headless) {
  using namespace grassland::rhi;
  if (headless) {
    std::cout << "imgui: skipped (requires a window for platform input)\n";
    return 0;
  }
  auto core = grassland::rhix::CreateCore(api);
  if (!core)
    return 1;
  auto device = core.CreateDeviceByAutoSelection();
  if (!device)
    return 1;
  auto queue = device.GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue)
    return 1;
  PresentMode requested = PRESENT_MODE_VSYNC;
  bool present_mode_changed = false;
  bool record_ui = true;
  auto window = device.CreateGLFWWindow({1280, 720, "ImGui - I: toggle UI recording; P: VSync"});
  if (!window)
    return 1;
  if (!window.EnableImGui())
    return 1;
  window.KeyEvent().RegisterCallback([&](int key, int, int action, int) {
    if (action != GLFW_PRESS)
      return;
    if (key == GLFW_KEY_ESCAPE)
      glfwSetWindowShouldClose(window.GLFWWindow(), GLFW_TRUE);
    if (key == GLFW_KEY_P) {
      requested = requested == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
      present_mode_changed = true;
    }
    if (key == GLFW_KEY_I) {
      record_ui = !record_ui;
      std::cout << "ImGui recording: " << std::boolalpha << record_ui << '\n';
    }
  });
  window.ImGuiEvent().RegisterCallback([&, clicks = 0, show_demo = true]() mutable {
    // Keep the context enabled: an empty frame must not replay previous UI.
    if (!record_ui)
      return;
    ImGui::Begin("ImGui recording");
    ImGui::TextUnformatted("I: toggle UI recording; P: VSync; Escape: close");
    if (ImGui::Button("Click me"))
      ++clicks;
    ImGui::Text("Clicks: %d", clicks);
    ImGui::Checkbox("Show ImGui demo", &show_demo);
    ImGui::End();
    if (show_demo)
      ImGui::ShowDemoWindow(&show_demo);
  });
  grassland::FPSCounter fps;
  auto title_time = std::chrono::steady_clock::now();
  for (int rendered = 0; !frames || rendered < frames;) {
    if (!grassland::rhix::Window::PollEvents())
      return 1;
    if (window.ShouldClose())
      break;
    if (!window.UpdateImGui())
      return 1;
    if (window.IsSwapChainInvalid()) {
      if (!window.RebuildSwapChain())
        return 1;
    }
    if (present_mode_changed) {
      present_mode_changed = false;
      if (!window.SetPresentMode(requested))
        return 1;
    }
    auto image = window.GetNextImage();
    if (!image) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    auto commands = device.CreateCommandContext();
    if (!commands)
      return 1;
    if (!commands.CmdClearColor(image, {0.6f, 0.7f, 0.8f, 1.0f}))
      return 1;
    if (!commands.CmdPresent(window))
      return 1;
    auto done = queue.Submit(std::move(commands));
    if (!done)
      return 1;
    ++rendered;
    fps.TickFrame();
    auto now = std::chrono::steady_clock::now();
    if (now - title_time >= std::chrono::milliseconds(250)) {
      if (!window.SetTitle(fmt::format("ImGui - FPS: {:.1f} - UI recording: {} - I: toggle", fps.GetFPS(), record_ui)))
        return 1;
      title_time = now;
    }
  }
  return 0;
}
