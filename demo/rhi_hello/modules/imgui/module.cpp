#include "../../module.h"

#include <fmt/format.h>

#include <chrono>
#include <iostream>
#include <thread>

#include "../../../rhi_common/last_frame.h"
#include "grassland/rhi/rhi.h"
#include "grassland/util/fps_counter.h"
#include "imgui.h"

int RunImGui(grassland::rhi::BackendAPI api,
             int frames,
             bool headless,
             const char *font_file,
             float font_size,
             const std::string &capture_directory) {
  using namespace grassland::rhi;
  if (headless) {
    std::cout << "imgui: skipped (requires a window for platform input)\n";
    return 0;
  }
  ObjPtr<ICore> core = CreateCore(api);
  if (!core)
    return 1;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  if (!device)
    return 1;
  rhi_demo::LastFrame capture(capture_directory, "imgui");
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue)
    return 1;
  PresentMode requested = PRESENT_MODE_VSYNC;
  bool present_mode_changed = false;
  bool record_ui = true;
  ObjPtr<IWindow> window = device->CreateGLFWWindow({1280, 720, "ImGui - I: toggle UI recording; P: VSync"});
  if (!window)
    return 1;
  if (!window->EnableImGui(true, font_file, font_size))
    return 1;
  window->KeyEvent().RegisterCallback([&](int key, int, int action, int) {
    if (action != GLFW_PRESS)
      return;
    if (key == GLFW_KEY_ESCAPE)
      glfwSetWindowShouldClose(window->GLFWWindow(), GLFW_TRUE);
    if (key == GLFW_KEY_P) {
      requested = requested == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
      present_mode_changed = true;
    }
    if (key == GLFW_KEY_I) {
      record_ui = !record_ui;
      std::cout << "ImGui recording: " << std::boolalpha << record_ui << '\n';
    }
  });
  int clicks = 0;
  bool show_demo = true;
  grassland::FPSCounter fps;
  auto title_time = std::chrono::steady_clock::now();
  for (int rendered = 0; !frames || rendered < frames;) {
    if (!PollEvents())
      return 1;
    if (window->ShouldClose())
      break;
    if (!window->UpdateImGui([&] {
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
        }))
      return 1;
    if (window->IsSwapChainInvalid()) {
      if (!window->RebuildSwapChain())
        return 1;
    }
    if (present_mode_changed) {
      present_mode_changed = false;
      if (!window->SetPresentMode(requested))
        return 1;
    }
    ObjPtr<IImage> image = window->GetNextImage();
    if (!image) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    ObjPtr<ICommandContext> commands = device->CreateCommandContext();
    if (!commands)
      return 1;
    if (!commands->CmdClearColor(image.Get(), {0.6f, 0.7f, 0.8f, 1.0f}))
      return 1;
    if (!commands->CmdPresent(window.Get()))
      return 1;
    if (!capture.Record(commands.Get(), image.Get(), window.Get()))
      return 1;
    ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());

    if (!done)
      return 1;
    ++rendered;
    fps.TickFrame();
    auto now = std::chrono::steady_clock::now();
    if (now - title_time >= std::chrono::milliseconds(250)) {
      if (!window->SetTitle(fmt::format("ImGui - FPS: {:.1f} - UI recording: {} - I: toggle", fps.GetFPS(), record_ui)))
        return 1;
      title_time = now;
    }
  }
  if (!capture.Save())
    return 1;
  return 0;
}
