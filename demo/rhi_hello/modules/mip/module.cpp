#include <chrono>
#include <iostream>
#include <thread>

#include "imgui.h"
#include "scene.h"

int RunMip(grassland::rhi::IDevice *device) {
  using namespace grassland::rhi;
  if (!device->GetPhysicalDevice().format_features[IMAGE_FORMAT_R8G8B8A8_UNORM].mip_compute_generation ||
      device->GetPhysicalDevice().mip_feature.max_anisotropy < 2) {
    std::cerr << "The mip demo requires GPU mip generation and anisotropic filtering.\n";
    return 1;
  }
  ObjPtr<IWindow> window = device->CreateGLFWWindow({1280, 960, "RHI mip - Space: pause; Escape: close"});
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  if (!window || !queue || !window->EnableImGui())
    return 1;
  rhi_mip_demo::Scene scene;
  {
    ObjPtr<IImage> frame = window->GetNextImage();
    if (!frame || !scene.Initialize(device, frame->GetDesc().format))
      return 1;
  }
  bool paused = false;
  window->KeyEvent().RegisterCallback([&](int key, int, int action, int) {
    if (action != GLFW_PRESS)
      return;
    if (key == GLFW_KEY_ESCAPE)
      glfwSetWindowShouldClose(window->GLFWWindow(), GLFW_TRUE);
    if (key == GLFW_KEY_SPACE)
      paused = !paused;
  });
  auto previous = std::chrono::steady_clock::now();
  float seconds = 0;
  while (true) {
    if (!PollEvents())
      return 1;
    if (window->ShouldClose())
      break;
    auto now = std::chrono::steady_clock::now();
    if (!paused)
      seconds += std::chrono::duration<float>(now - previous).count();
    previous = now;
    if (!window->UpdateImGui([&] {
          auto *draw = ImGui::GetBackgroundDrawList();
          const auto size = ImGui::GetIO().DisplaySize;
          const char *titles[] = {"NO MIP / bilinear", "TRILINEAR", "ANISOTROPIC"};
          for (int col = 0; col < 3; ++col) {
            const float x = (col + 0.5f) * size.x / 3;
            const auto text = ImGui::CalcTextSize(titles[col]);
            draw->AddText({x - text.x / 2, 18}, IM_COL32(235, 240, 255, 255), titles[col]);
          }
          char details[128];
          std::snprintf(details, sizeof(details),
                        "2048 x 2048 | GPU mip chain | Anisotropy %ux | Space: %s | Escape: close", scene.Anisotropy(),
                        paused ? "resume" : "pause");
          draw->AddText({18, 44}, IM_COL32(165, 180, 205, 255), details);
          const char *rows[] = {"SMALL", "MEDIUM", "LARGE"};
          for (int row = 0; row < 3; ++row)
            draw->AddText({18, 80 + row * (size.y - 72) / 3}, IM_COL32(150, 170, 195, 255), rows[row]);
        }))
      return 1;
    if (window->IsSwapChainInvalid() && !window->RebuildSwapChain())
      return 1;
    ObjPtr<IImage> frame = window->GetNextImage();
    if (!frame) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    ObjPtr<ICommandContext> commands = queue->CreateCommandContext();
    if (!commands || !scene.Draw(commands.Get(), frame.Get(), seconds) || !commands->CmdPresent(window.Get()))
      return 1;
    ObjPtr<ISubmissionToken> done = commands->Submit();
    if (!done)
      return 1;
  }
  return 0;
}
