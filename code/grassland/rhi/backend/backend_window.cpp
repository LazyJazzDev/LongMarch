#include "grassland/rhi/backend/backend_window.h"

#include <algorithm>
#include <cstdlib>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

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

bool Window::GetHDRReferenceWhiteNits(float &nits) const {
#ifdef _WIN32
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return false;
#endif
  MONITORINFOEXW monitor{};
  monitor.cbSize = sizeof(monitor);
  if (!GetMonitorInfoW(MonitorFromWindow(glfwGetWin32Window(window_), MONITOR_DEFAULTTONEAREST), &monitor))
    return false;
  UINT32 path_count{}, mode_count{};
  if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &path_count, &mode_count) != ERROR_SUCCESS)
    return false;
  std::vector<DISPLAYCONFIG_PATH_INFO> paths(path_count);
  std::vector<DISPLAYCONFIG_MODE_INFO> modes(mode_count);
  if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &path_count, paths.data(), &mode_count, modes.data(), nullptr) !=
      ERROR_SUCCESS)
    return false;
  for (UINT32 i = 0; i < path_count; ++i) {
    const auto &path = paths[i];
    DISPLAYCONFIG_SOURCE_DEVICE_NAME source{};
    source.header = {DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME, sizeof(source), path.sourceInfo.adapterId,
                     path.sourceInfo.id};
    if (DisplayConfigGetDeviceInfo(&source.header) != ERROR_SUCCESS ||
        wcscmp(source.viewGdiDeviceName, monitor.szDevice))
      continue;
    DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO color{};
    color.header = {DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO, sizeof(color), path.targetInfo.adapterId,
                    path.targetInfo.id};
    if (DisplayConfigGetDeviceInfo(&color.header) != ERROR_SUCCESS || !color.advancedColorEnabled)
      return false;
    DISPLAYCONFIG_SDR_WHITE_LEVEL white{};
    white.header = {DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL, sizeof(white), path.targetInfo.adapterId,
                    path.targetInfo.id};
    if (DisplayConfigGetDeviceInfo(&white.header) != ERROR_SUCCESS || !white.SDRWhiteLevel)
      return false;
    nits = 80.0f * (white.SDRWhiteLevel / 1000.0f);
    return true;
  }
#endif
  return false;
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
