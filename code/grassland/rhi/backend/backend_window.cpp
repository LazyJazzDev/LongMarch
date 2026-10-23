#include "grassland/rhi/backend/backend_window.h"

#include <algorithm>
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
  const auto options = GetSupportedPresentationOptions();
  if (std::none_of(options.begin(), options.end(),
                   [&](const auto &option) { return option.color_space == requested_color_space_; })) {
    ReportError(-1, "Requested swapchain color space is unavailable.");
    return false;
  }
  invalid_ = true;
  ReleaseNextImage();
  if (!RebuildNativeSwapChain(requested_mode_))
    return false;
  int width, height;
  glfwGetFramebufferSize(window_, &width, &height);
  if (glfwGetWindowAttrib(window_, GLFW_ICONIFIED) || width <= 0 || height <= 0)
    return true;
  color_space_ = requested_color_space_;
  invalid_ = false;
  if (UpdateNextImage())
    return true;
  invalid_ = true;
  return false;
}

bool Window::SetPresentMode(PresentMode mode) {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return false;
#endif
  requested_mode_ = mode;
  return RebuildSwapChain();
}

bool Window::SetColorSpace(ColorSpace color_space) {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return false;
#endif
  const auto options = GetSupportedPresentationOptions();
  if (std::none_of(options.begin(), options.end(),
                   [&](const auto &option) { return option.color_space == color_space; }))
    return false;
  const auto previous = requested_color_space_;
  requested_color_space_ = color_space;
  if (RebuildSwapChain())
    return true;
  requested_color_space_ = previous;
  return false;
}

ColorSpace Window::GetColorSpace() const {
  return color_space_;
}

ColorSpace Window::RequestedColorSpace() const {
  return requested_color_space_;
}

Window::~Window() {
  glfw::UnregisterWindow(window_);
  glfw::DestroyWindow(window_);
}

bool Window::InitializeWindow(const WindowDesc &desc) {
  requested_mode_ = desc.present_mode;
  requested_color_space_ = desc.color_space;
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
