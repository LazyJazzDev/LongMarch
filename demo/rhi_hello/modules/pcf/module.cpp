#include <chrono>
#include <thread>

#include "imgui.h"
#include "scene.h"

int RunPCF(grassland::rhi::IDevice *device) {
  using namespace grassland::rhi;
  ObjPtr<IWindow> window = device->CreateGLFWWindow({1440, 900, "RHI pcf - Left: hard shadows | Right: PCF"});
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  if (!window || !queue || !window->EnableImGui())
    return 1;
  rhi_pcf_demo::Scene scene;
  {
    ObjPtr<IImage> frame = window->GetNextImage();
    if (!frame || !scene.Initialize(device, frame->GetDesc().format))
      return 1;
  }
  bool visibility_only = false;
  window->KeyEvent().RegisterCallback([&](int key, int, int action, int) {
    if (action != GLFW_PRESS)
      return;
    if (key == GLFW_KEY_ESCAPE)
      glfwSetWindowShouldClose(window->GLFWWindow(), GLFW_TRUE);
    if (key == GLFW_KEY_SPACE)
      visibility_only = !visibility_only;
  });
  while (true) {
    if (!PollEvents())
      return 1;
    if (window->ShouldClose())
      break;
    if (!window->UpdateImGui([&] {
          auto *draw = ImGui::GetBackgroundDrawList();
          const auto size = ImGui::GetIO().DisplaySize;
          const char *labels[] = {"HARD SHADOW / single depth test", "PCF / 5 x 5 filtered comparisons"};
          for (int side = 0; side < 2; ++side) {
            const auto text = ImGui::CalcTextSize(labels[side]);
            draw->AddText({(side + .5f) * size.x / 2 - text.x / 2, 18}, IM_COL32(240, 230, 210, 255), labels[side]);
          }
          draw->AddText(
              {18, 44}, IM_COL32(170, 185, 205, 255),
              "Same light, geometry, 512 x 512 depth map and bias | Space: shadow visibility | Escape: close");
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
    if (!commands || !scene.Draw(commands.Get(), frame.Get(), visibility_only) || !commands->CmdPresent(window.Get()))
      return 1;
    ObjPtr<ISubmissionToken> done = commands->Submit();
    if (!done)
      return 1;
  }
  return 0;
}
