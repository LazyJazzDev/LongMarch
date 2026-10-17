#include "grassland/rhi/backend/backend_window.h"

#include <cstdlib>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace grassland::rhi::backend {

Window::Window() = default;

rhi::IImage *Window::GetNextImage() {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread()) {
    ReportError(-1, "GetNextImage must run on the window thread.");
    std::abort();
  }
#endif
  if (invalid_)
    return nullptr;
  int width, height;
  glfwGetFramebufferSize(window_, &width, &height);
  if (glfwGetWindowAttrib(window_, GLFW_ICONIFIED) || width <= 0 || height <= 0)
    return nullptr;
  std::lock_guard<std::mutex> lock(next_image_mutex_);
#if !defined(NDEBUG)
  if (!next_image_) {
    ReportError(-1, "No prepared swapchain image for a drawable window.");
    std::abort();
  }
#endif
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
  // A native OUT_OF_DATE result defers recovery to an explicit rebuild.
  if (next_image_ || invalid_)
    return true;
  invalid_ = true;
  return false;
}

void Window::ReleaseNextImage() {
  std::lock_guard<std::mutex> lock(next_image_mutex_);
  next_image_.Reset();
}

bool Window::IsSwapChainInvalid() const {
  return invalid_;
}

bool Window::RebuildSwapChain() {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return false;
#endif
  try {
    invalid_ = true;
    ReleaseNextImage();
    if (!RebuildNativeSwapChain(requested_mode_))
      return false;
    int width, height;
    glfwGetFramebufferSize(window_, &width, &height);
    if (glfwGetWindowAttrib(window_, GLFW_ICONIFIED) || width <= 0 || height <= 0)
      return true;
    invalid_ = false;
    if (UpdateNextImage())
      return true;
    invalid_ = true;
    return false;
  } catch (const std::bad_alloc &) {
    invalid_ = true;
    ReportError(-1, "Swapchain rebuild ran out of memory.");
    return false;
  }
}

bool Window::SetPresentMode(PresentMode mode) {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return false;
#endif
  requested_mode_ = mode;
  return RebuildSwapChain();
}

Window::~Window() {
  glfw::UnregisterWindow(window_);
  glfw::DestroyWindow(window_);
}

bool Window::InitializeWindow(const WindowDesc &desc) {
  requested_mode_ = desc.present_mode;
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
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return true;
#endif
  return glfwWindowShouldClose(window_);
}

bool Window::IsKeyDown(int key) const {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return false;
#endif
  return glfwGetKey(window_, key) == GLFW_PRESS;
}

bool Window::SetTitle(const std::string &title) {
  return glfw::SetWindowTitle(window_, title.c_str());
}

}  // namespace grassland::rhi::backend
