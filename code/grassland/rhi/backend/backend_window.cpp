#include "grassland/rhi/backend/backend_window.h"

#include <cstdlib>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace grassland::rhi::backend {

Window::Window() = default;

rhi::IImage *Window::GetNextImage() {
  if (!glfw::IsWindowThread()) {
    ReportError(-1, "GetNextImage must run on the window thread.");
    std::abort();
  }
  int width, height;
  glfwGetFramebufferSize(window_, &width, &height);
  if (glfwGetWindowAttrib(window_, GLFW_ICONIFIED) || width <= 0 || height <= 0)
    return nullptr;
  std::lock_guard<std::mutex> lock(next_image_mutex_);
  if (!next_image_) {
    ReportError(-1, "No prepared swapchain image for a drawable window.");
    std::abort();
  }
  return ObjectPtr<Image>(next_image_).Detach();
}

bool Window::UpdateNextImage() {
  auto image = AcquireNativeFrame();
  std::lock_guard<std::mutex> lock(next_image_mutex_);
  next_image_ = std::move(image);
  int width, height;
  glfwGetFramebufferSize(window_, &width, &height);
  if (glfwGetWindowAttrib(window_, GLFW_ICONIFIED) || width <= 0 || height <= 0)
    return true;
  return bool(next_image_);
}

void Window::ReleaseNextImage() {
  std::lock_guard<std::mutex> lock(next_image_mutex_);
  next_image_.Reset();
}

bool Window::RebuildSwapChain() {
  if (!glfw::IsWindowThread())
    return false;
  try {
    ReleaseNextImage();
    if (!RebuildNativeSwapChain())
      return false;
    return UpdateNextImage();
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Swapchain rebuild ran out of memory.");
    return false;
  }
}

bool Window::SetPresentMode(PresentMode mode) {
  try {
    if (!glfw::IsWindowThread())
      return false;
    if (!SetNativePresentMode(mode))
      return false;
    return RebuildSwapChain();
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Present mode change ran out of memory.");
    return false;
  }
}

Window::~Window() {
  glfw::UnregisterWindow(window_);
  glfw::DestroyWindow(window_);
}

bool Window::InitializeWindow(const WindowDesc &desc) {
  window_ = glfw::CreateGLFWWindow(desc.width, desc.height, desc.title.c_str());
  if (!window_)
    return false;
  glfw::RegisterWindow(this);
  return true;
}

GLFWwindow *Window::GLFWWindow() const {
  return window_;
}

bool Window::ShouldClose() const {
  return !glfw::IsWindowThread() || glfwWindowShouldClose(window_);
}

bool Window::IsKeyDown(int key) const {
  return glfw::IsWindowThread() && glfwGetKey(window_, key) == GLFW_PRESS;
}

bool Window::SetTitle(const std::string &title) {
  return glfw::SetWindowTitle(window_, title.c_str());
}

}  // namespace grassland::rhi::backend
