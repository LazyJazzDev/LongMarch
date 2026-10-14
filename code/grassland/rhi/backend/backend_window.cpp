#include "grassland/rhi/backend/backend_window.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace grassland::rhi::backend {

IWindow::IWindow() = default;

IWindow::~IWindow() {
  glfw::DestroyWindow(window_);
}

bool IWindow::InitializeWindow(const WindowDesc &desc) {
  window_ = glfw::CreateWindow(desc.width, desc.height, desc.title.c_str());
  return window_ != nullptr;
}

GLFWwindow *IWindow::NativeWindow() const {
  return window_;
}

bool IWindow::ShouldClose() const {
  return !glfw::IsWindowThread() || glfwWindowShouldClose(window_);
}

bool IWindow::IsKeyDown(Key key) const {
  return glfw::IsWindowThread() && glfwGetKey(window_, key == Key::P ? GLFW_KEY_P : GLFW_KEY_ESCAPE) == GLFW_PRESS;
}

}  // namespace grassland::rhi::backend
