#include "grassland/rhi/backend/backend_window.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "grassland/rhi/backend/backend_window_imgui.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

namespace grassland::rhi::backend {

Window::Window() = default;

bool Window::EnableImGui(bool enabled, const char *font_file_path, float font_size) {
  if (!glfw::IsWindowThread() || processing_imgui_)
    return false;
  if (!enabled) {
    imgui_.reset();
    return true;
  }
  if (!imgui_)
    imgui_ = WindowImGui::Create(device_, window_, font_file_path, font_size);
  return bool(imgui_);
}

ImGuiContext *Window::GetImGuiContext() const {
  return imgui_ ? imgui_->Context() : nullptr;
}

bool Window::UpdateImGui() {
  if (!imgui_)
    return true;
  if (!glfw::IsWindowThread() || processing_imgui_)
    return false;

  struct ProcessingScope {
    bool &flag;

    ProcessingScope(bool &value) : flag(value) {
      flag = true;
    }

    ~ProcessingScope() {
      flag = false;
    }
  } scope(processing_imgui_);

  return imgui_->Update(imgui_event_);
}

bool Window::RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image) {
  if (!imgui_)
    return true;
  if (!glfw::IsWindowThread() || processing_imgui_)
    return false;

  struct ProcessingScope {
    bool &flag;

    ProcessingScope(bool &value) : flag(value) {
      flag = true;
    }

    ~ProcessingScope() {
      flag = false;
    }
  } scope(processing_imgui_);

  return imgui_->Record(commands, image);
}

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

float Window::GetHDRReferenceWhiteScale() const {
#ifdef _WIN32
  float nits;
  return GetHDRReferenceWhiteNits(nits) ? nits / 80.0f : 1.0f;
#else
  // Linux uses fixed source encoding parameters; macOS uses relative EDR.
  return 1.0f;
#endif
}

#ifndef __APPLE__
float Window::GetHDRHeadroom() const {
#ifdef _WIN32
  float white;
  if (!GetHDRReferenceWhiteNits(white))
    return 10.0f;
  const auto monitor = MonitorFromWindow(glfwGetWin32Window(window_), MONITOR_DEFAULTTONEAREST);
  Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
  if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
    return 10.0f;
  for (UINT a = 0;; ++a) {
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    if (factory->EnumAdapters1(a, &adapter) != S_OK)
      break;
    for (UINT o = 0;; ++o) {
      Microsoft::WRL::ComPtr<IDXGIOutput> output;
      if (adapter->EnumOutputs(o, &output) != S_OK)
        break;
      Microsoft::WRL::ComPtr<IDXGIOutput6> output6;
      DXGI_OUTPUT_DESC1 desc{};
      if (FAILED(output.As(&output6)) || FAILED(output6->GetDesc1(&desc)) || desc.Monitor != monitor)
        continue;
      const float ratio = desc.MaxLuminance / white;
      return std::isfinite(ratio) && ratio > 0.0f ? std::max(1.0f, ratio) : 10.0f;
    }
  }
#endif
  return 10.0f;
}

#endif

bool Window::GetHDRReferenceWhiteNits(float &nits) const {
  if (!glfw::IsWindowThread())
    return false;
#ifdef _WIN32
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
  imgui_.reset();
  glfw::UnregisterWindow(window_);
  glfw::DestroyWindow(window_);
}

bool Window::InitializeWindow(Device *device, const WindowDesc &desc) {
  device_ = device;
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
