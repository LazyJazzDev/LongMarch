#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/backend_window.h"

namespace grassland::rhi::backend {
struct WindowState final {
  WindowState();
  ~WindowState();
  bool invalid_{true};
  IBackendDevice *device_{};
  IBackendQueue *queue_{};
  std::unique_ptr<WindowImGui> imgui_{};
  std::unique_ptr<ImagePresenter> presenter_{};
  bool processing_imgui_{};
  EventManager<void(int, int)> resize_event_{};
  EventManager<void(int, int)> framebufferresize_event_{};
  EventManager<void(bool)> cursorenter_event_{};
  EventManager<void(bool)> focus_event_{};
  EventManager<void(double, double)> mousemove_event_{};
  EventManager<void(int, int, int, double, double)> mousebutton_event_{};
  EventManager<void(double, double)> scroll_event_{};
  EventManager<void(int, int, int, int)> key_event_{};
  EventManager<void(uint32_t)> char_event_{};
  EventManager<void(int, const char **)> drop_event_{};
  PresentMode requested_mode_{PRESENT_MODE_VSYNC};
  ColorSpace requested_color_space_{COLOR_SPACE_SRGB_NONLINEAR};
  ColorSpace color_space_{COLOR_SPACE_SRGB_NONLINEAR};
  GLFWwindow *window_{};
  std::mutex next_image_mutex_{};
  ObjectPtr<IBackendImage> next_image_{};
};
}  // namespace grassland::rhi::backend
