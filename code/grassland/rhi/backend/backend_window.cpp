#include "grassland/rhi/backend/backend_window.h"

#include <cassert>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace grassland::rhi::backend {

IWindow::IWindow() = default;

void IWindow::AcquireFrameLease() {
  frame_leases_.fetch_add(1, std::memory_order_relaxed);
}

void IWindow::ReleaseFrameLease() {
  frame_leases_.fetch_sub(1, std::memory_order_release);
}

bool IWindow::HasFrameLeases() const {
  return frame_leases_.load(std::memory_order_acquire) != 0;
}

IWindow::~IWindow() {
  assert(!HasFrameLeases());
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

bool IWindow::SetTitle(const std::string &title) {
  return glfw::SetWindowTitle(window_, title.c_str());
}

}  // namespace grassland::rhi::backend
