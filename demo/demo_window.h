#pragma once

#include <glm/glm.hpp>

#include "grassland/rhi/rhi.h"

namespace demo {

// Demo input conveniences around an owned RHI window. Rendering and all event
// delivery, including native gestures, remain in RHI/GLFW.
class Window {
 public:
  explicit Window(grassland::rhi::IWindow *window);
  ~Window();
  Window(const Window &) = delete;
  Window &operator=(const Window &) = delete;

  grassland::rhi::IWindow *Get() const {
    return window_.Get();
  }

  GLFWwindow *GLFWWindow() const {
    return window_->GLFWWindow();
  }

  bool ShouldClose() const {
    return window_->ShouldClose();
  }

  bool IsKeyDown(int key) const {
    return window_->IsKeyDown(key);
  }

  bool SetTitle(const std::string &title) {
    return window_->SetTitle(title);
  }

  glm::ivec2 GetSize() const;
  glm::ivec2 GetFramebufferSize() const;
  glm::dvec2 GetCursorPosition() const;

  void RequestClose() {
    glfwSetWindowShouldClose(GLFWWindow(), GLFW_TRUE);
  }

  void Focus() {
    glfwFocusWindow(GLFWWindow());
  }

  auto &KeyEvent() {
    return window_->KeyEvent();
  }

  auto &MouseMoveEvent() {
    return window_->MouseMoveEvent();
  }

  auto &MouseButtonEvent() {
    return window_->MouseButtonEvent();
  }

  auto &CursorEnterEvent() {
    return window_->CursorEnterEvent();
  }

  auto &FocusEvent() {
    return window_->FocusEvent();
  }

  auto &ScrollEvent() {
    return window_->ScrollEvent();
  }

  auto &MagnifyEvent() {
    return window_->MagnifyEvent();
  }

 private:
  grassland::rhi::ObjPtr<grassland::rhi::IWindow> window_;
};

}  // namespace demo
