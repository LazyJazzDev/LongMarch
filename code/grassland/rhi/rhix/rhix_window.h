#pragma once
#include "grassland/rhi/rhi.h"
#include "grassland/rhi/rhix/rhix_image.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
class Window : public ObjectContainer {
 public:
  Window(rhi::IWindow *object) noexcept;
  rhi::IWindow *Get() const noexcept;
  Window() noexcept;
  // Window creation, events, acquisition and presentation run on the main thread.
  static bool PollEvents();
  GLFWwindow *GLFWWindow() const;
  bool ShouldClose() const;
  bool IsKeyDown(int key) const;
  bool SetTitle(const std::string &title) const;
  Image GetNextImage() const;
  bool SetPresentMode(rhi::PresentMode mode) const;
  rhi::PresentMode GetPresentMode() const;
  std::vector<rhi::WindowPresentationOption> GetSupportedPresentationOptions() const;
};

}  // namespace grassland::rhix
