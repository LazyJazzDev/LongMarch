#include <chrono>
#include <thread>

#include "grassland/rhi/rhi.h"

int RunWindow(grassland::rhi::IDevice *device) {
  using namespace grassland::rhi;
  ObjPtr<IWindow> window = device->CreateGLFWWindow({1280, 720, "RHI window - Escape: close"});
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!window || !queue)
    return 1;
  window->KeyEvent().RegisterCallback([&](int key, int, int action, int) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
      glfwSetWindowShouldClose(window->GLFWWindow(), GLFW_TRUE);
  });
  while (true) {
    if (!PollEvents())
      return 1;
    if (window->ShouldClose())
      break;
    if (window->IsSwapChainInvalid() && !window->RebuildSwapChain())
      return 1;
    ObjPtr<IImage> image = window->GetNextImage();
    if (!image) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    ObjPtr<ICommandContext> commands = device->CreateCommandContext();
    if (!commands || !commands->CmdClearImage(image.Get(), {0.6f, 0.7f, 0.8f, 1.0f}) ||
        !commands->CmdPresent(window.Get()))
      return 1;
    // Submit borrows commands; accepted GPU work retains its resources until completion.
    ObjPtr<ISubmissionToken> done = queue->Submit(commands.Get());
    if (!done)
      return 1;
  }
  return 0;
}
