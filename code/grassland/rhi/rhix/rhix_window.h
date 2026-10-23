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
  bool EnableImGui(bool enabled = true) const;
  ImGuiContext *GetImGuiContext() const;
  EventManager<void(int, int)> &ResizeEvent() const;
  EventManager<void(int, int)> &FramebufferResizeEvent() const;
  EventManager<void(bool)> &CursorEnterEvent() const;
  EventManager<void(bool)> &FocusEvent() const;
  EventManager<void(double, double)> &MouseMoveEvent() const;
  EventManager<void(int, int, int, double, double)> &MouseButtonEvent() const;
  EventManager<void(double, double)> &ScrollEvent() const;
  EventManager<void(int, int, int, int)> &KeyEvent() const;
  EventManager<void(uint32_t)> &CharEvent() const;
  EventManager<void(int, const char **)> &DropEvent() const;
  EventManager<void()> &ImGuiEvent() const;
  Image GetNextImage() const;
  bool IsSwapChainInvalid() const;
  bool RebuildSwapChain() const;
  bool SetPresentMode(rhi::PresentMode mode) const;
  rhi::PresentMode GetPresentMode() const;
  std::vector<rhi::WindowPresentationOption> GetSupportedPresentationOptions() const;
};

}  // namespace grassland::rhix
