#pragma once
#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>

#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
class IImage;
// Window operations and presentation run on the main thread.
bool PollEvents();

class IWindow : public IObject {
 public:
  // Borrowed handle; Window owns it. Use GLFW operations on the main thread.
  virtual GLFWwindow *GLFWWindow() const = 0;
  virtual bool ShouldClose() const = 0;
  virtual bool IsKeyDown(int key) const = 0;
  virtual bool SetTitle(const std::string &title) = 0;
  // Never rebuilds the swapchain. Returns nullptr when no frame is available.
  virtual IImage *AcquireFrame() = 0;
  // Explicit rebuild; call on the window thread after size changes or invalidation.
  virtual bool RebuildSwapChain() = 0;
  virtual bool SetPresentMode(PresentMode mode) = 0;
  virtual PresentMode GetPresentMode() const = 0;
  // Currently implemented combinations supported by this window's surface.
  // Formats are selected automatically at creation/rebuild. Empty on failure.
  virtual std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const = 0;

 protected:
  ~IWindow() override;
};

}  // namespace grassland::rhi
