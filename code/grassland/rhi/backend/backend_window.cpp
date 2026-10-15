#include "grassland/rhi/backend/backend_window.h"

#include <cassert>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace grassland::rhi::backend {

Window::Window() = default;

rhi::IImage *Window::AcquireFrame() {
  try {
    if (glfw::IsWindowThread())
      return AcquireNativeFrame().Detach();
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Frame acquisition ran out of memory.");
  }
  return nullptr;
}

bool Window::SetPresentMode(PresentMode mode) {
  try {
    return glfw::IsWindowThread() && SetNativePresentMode(mode);
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Present mode change ran out of memory.");
    return false;
  }
}

void Window::AcquireFrameLease() {
  frame_leases_.fetch_add(1, std::memory_order_relaxed);
}

void Window::ReleaseFrameLease() {
  frame_leases_.fetch_sub(1, std::memory_order_release);
}

bool Window::HasFrameLeases() const {
  return frame_leases_.load(std::memory_order_acquire) != 0;
}

Window::~Window() {
  assert(!HasFrameLeases());
  glfw::DestroyWindow(window_);
}

bool Window::InitializeWindow(const WindowDesc &desc) {
  window_ = glfw::CreateGLFWWindow(desc.width, desc.height, desc.title.c_str());
  return window_ != nullptr;
}

GLFWwindow *Window::NativeWindow() const {
  return window_;
}

bool Window::ShouldClose() const {
  return !glfw::IsWindowThread() || glfwWindowShouldClose(window_);
}

bool Window::IsKeyDown(Key key) const {
  return glfw::IsWindowThread() && glfwGetKey(window_, key == Key::P ? GLFW_KEY_P : GLFW_KEY_ESCAPE) == GLFW_PRESS;
}

bool Window::SetTitle(const std::string &title) {
  return glfw::SetWindowTitle(window_, title.c_str());
}

}  // namespace grassland::rhi::backend
