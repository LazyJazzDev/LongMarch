#include "grassland/rhi/backend/backend_window.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_image_presenter.h"
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_window_imgui.h"
#include "grassland/rhi/backend/backend_window_state.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

namespace grassland::rhi::backend {

void IBackendWindow::InitializeWindow() {
}

bool IBackendWindow::CommonEnableImGui(bool enabled, const char *font_file_path, float font_size) {
  if (!glfw::IsWindowThread() || StateWindow().processing_imgui_)
    return false;
  if (!enabled) {
    StateWindow().imgui_.reset();
    return true;
  }
  if (!StateWindow().imgui_)
    StateWindow().imgui_ = WindowImGui::Create(StateWindow().device_, StateWindow().window_, font_file_path, font_size);
  return bool(StateWindow().imgui_);
}

ImGuiContext *IBackendWindow::GetImGuiContext() const {
  return StateWindow().imgui_ ? StateWindow().imgui_->Context() : nullptr;
}

bool IBackendWindow::CommonUpdateImGui(const std::function<void()> &callback) {
  if (!StateWindow().imgui_)
    return true;
  if (!glfw::IsWindowThread() || StateWindow().processing_imgui_)
    return false;

  struct ProcessingScope {
    bool &flag;

    ProcessingScope(bool &value) : flag(value) {
      flag = true;
    }

    ~ProcessingScope() {
      flag = false;
    }
  } scope(StateWindow().processing_imgui_);

  return StateWindow().imgui_->Update(callback);
}

const void *IBackendWindow::DeviceIdentity() const {
  return StateWindow().device_;
}

bool IBackendWindow::RecordPresentImage(ICommandContext *commands, IImage *image, ColorSpace color_space) {
  ObjPtr<IImage> frame = GetNextImage();
  if (!frame)
    return false;
  if (!StateWindow().presenter_)
    StateWindow().presenter_ = std::make_unique<ImagePresenter>(StateWindow().device_);
  return StateWindow().presenter_->Record(commands, image, frame.Get(), color_space, GetColorSpace(),
                                          GetHDRReferenceWhiteScale());
}

bool IBackendWindow::RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image) {
  return RecordImGui(commands, image, GetColorSpace(), GetHDRReferenceWhiteScale());
}

bool IBackendWindow::RecordImGui(rhi::ICommandContext *commands,
                                 rhi::IImage *image,
                                 ColorSpace color_space,
                                 float white_scale) {
  if (!StateWindow().imgui_)
    return true;
  if (!glfw::IsWindowThread() || StateWindow().processing_imgui_)
    return false;

  struct ProcessingScope {
    bool &flag;

    ProcessingScope(bool &value) : flag(value) {
      flag = true;
    }

    ~ProcessingScope() {
      flag = false;
    }
  } scope(StateWindow().processing_imgui_);

  return StateWindow().imgui_->Record(commands, image, color_space, white_scale);
}

rhi::IImage *IBackendWindow::CommonGetNextImage() {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread()) {
    ReportError(-1, "GetNextImage must run on the window thread.");
    std::abort();
  }
#endif
  if (StateWindow().invalid_)
    return nullptr;
  int width, height;
  glfwGetFramebufferSize(StateWindow().window_, &width, &height);
  if (glfwGetWindowAttrib(StateWindow().window_, GLFW_ICONIFIED) || width <= 0 || height <= 0)
    return nullptr;
  std::lock_guard<std::mutex> lock(StateWindow().next_image_mutex_);
#if !defined(NDEBUG)
  if (!StateWindow().next_image_) {
    ReportError(-1, "No prepared swapchain image for a drawable window.");
    std::abort();
  }
#endif
  return ObjectPtr<IBackendImage>(StateWindow().next_image_).Detach();
}

IBackendQueue *IBackendWindow::OwnerQueue() const {
  return StateWindow().queue_;
}

bool IBackendWindow::CommonUpdateNextImage() {
  auto image = AcquireNativeFrame();
  std::lock_guard<std::mutex> lock(StateWindow().next_image_mutex_);
  StateWindow().next_image_ = std::move(image);
  int width, height;
  glfwGetFramebufferSize(StateWindow().window_, &width, &height);
  if (glfwGetWindowAttrib(StateWindow().window_, GLFW_ICONIFIED) || width <= 0 || height <= 0)
    return true;
  // A native OUT_OF_DATE result defers recovery to an explicit rebuild.
  if (StateWindow().next_image_ || StateWindow().invalid_)
    return true;
  StateWindow().invalid_ = true;
  return false;
}

void IBackendWindow::ReleaseNextImage() {
  std::lock_guard<std::mutex> lock(StateWindow().next_image_mutex_);
  StateWindow().next_image_.Reset();
}

bool IBackendWindow::CommonIsSwapChainInvalid() const {
  return StateWindow().invalid_;
}

bool IBackendWindow::CommonRebuildSwapChain() {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return false;
#endif
  const auto options = GetSupportedPresentationOptions();
  if (std::none_of(options.begin(), options.end(),
                   [&](const auto &option) { return option.color_space == StateWindow().requested_color_space_; })) {
    ReportError(-1, "Requested swapchain color space is unavailable.");
    return false;
  }
  StateWindow().invalid_ = true;
  ReleaseNextImage();
  if (!RebuildNativeSwapChain(StateWindow().requested_mode_))
    return false;
  int width, height;
  glfwGetFramebufferSize(StateWindow().window_, &width, &height);
  if (glfwGetWindowAttrib(StateWindow().window_, GLFW_ICONIFIED) || width <= 0 || height <= 0)
    return true;
  StateWindow().color_space_ = StateWindow().requested_color_space_;
  StateWindow().invalid_ = false;
  if (UpdateNextImage())
    return true;
  StateWindow().invalid_ = true;
  return false;
}

bool IBackendWindow::CommonSetPresentMode(PresentMode mode) {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return false;
#endif
  StateWindow().requested_mode_ = mode;
  return RebuildSwapChain();
}

bool IBackendWindow::CommonSetColorSpace(ColorSpace color_space) {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return false;
#endif
  const auto options = GetSupportedPresentationOptions();
  if (std::none_of(options.begin(), options.end(),
                   [&](const auto &option) { return option.color_space == color_space; }))
    return false;
  const auto previous = StateWindow().requested_color_space_;
  StateWindow().requested_color_space_ = color_space;
  if (RebuildSwapChain())
    return true;
  StateWindow().requested_color_space_ = previous;
  return false;
}

ColorSpace IBackendWindow::CommonGetColorSpace() const {
  return StateWindow().color_space_;
}

float IBackendWindow::CommonGetHDRReferenceWhiteScale() const {
#ifdef _WIN32
  float nits;
  return GetHDRReferenceWhiteNits(nits) ? nits / 80.0f : 1.0f;
#else
  // Linux uses fixed source encoding parameters; macOS uses relative EDR.
  return 1.0f;
#endif
}

#ifndef __APPLE__
float IBackendWindow::CommonGetHDRHeadroom() const {
#ifdef _WIN32
  float white;
  if (!GetHDRReferenceWhiteNits(white))
    return 10.0f;
  const auto monitor = MonitorFromWindow(glfwGetWin32Window(StateWindow().window_), MONITOR_DEFAULTTONEAREST);
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

bool IBackendWindow::CommonGetHDRReferenceWhiteNits(float &nits) const {
  if (!glfw::IsWindowThread())
    return false;
#ifdef _WIN32
  MONITORINFOEXW monitor{};
  monitor.cbSize = sizeof(monitor);
  if (!GetMonitorInfoW(MonitorFromWindow(glfwGetWin32Window(StateWindow().window_), MONITOR_DEFAULTTONEAREST),
                       &monitor))
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

ColorSpace IBackendWindow::RequestedColorSpace() const {
  return StateWindow().requested_color_space_;
}

void IBackendWindow::DestroyWindow() {
  StateWindow().imgui_.reset();
  glfw::UnregisterWindow(StateWindow().window_);
  glfw::DestroyWindow(StateWindow().window_);
}

bool IBackendWindow::InitializeWindow(IBackendDevice *device, const WindowDesc &desc) {
  StateWindow().device_ = device;
  StateWindow().queue_ =
      rhi::QueryInterfacePtr<IBackendQueue>(ObjectPtr<IQueue>::Adopt(device->GetQueue(QUEUE_TYPE_MAIN)).Get()).Get();
  StateWindow().requested_mode_ = desc.present_mode;
  StateWindow().requested_color_space_ = desc.color_space;
  StateWindow().window_ = glfw::CreateGLFWWindow(desc.width, desc.height, desc.title.c_str());
  if (!StateWindow().window_)
    return false;
  glfw::RegisterWindow(this);
  return true;
}

GLFWwindow *IBackendWindow::CommonGLFWWindow() const {
  return StateWindow().window_;
}

bool IBackendWindow::CommonShouldClose() const {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return true;
#endif
  return glfwWindowShouldClose(StateWindow().window_);
}

bool IBackendWindow::CommonIsKeyDown(int key) const {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return false;
#endif
  return glfwGetKey(StateWindow().window_, key) == GLFW_PRESS;
}

bool IBackendWindow::CommonSetTitle(const std::string &title) {
  return glfw::SetWindowTitle(StateWindow().window_, title.c_str());
}

}  // namespace grassland::rhi::backend

LONGMARCH_RHI_UUID(grassland::rhi::backend::IBackendWindow, "093b830a-4e7b-4d76-8df4-9a2693735c9d")

namespace grassland::rhi::backend {
WindowState::WindowState() = default;
WindowState::~WindowState() = default;
}  // namespace grassland::rhi::backend

namespace grassland::rhi::backend {
EventManager<void(int, int)> &IBackendWindow::CommonResizeEvent() {
  return StateWindow().resize_event_;
}

EventManager<void(int, int)> &IBackendWindow::CommonFramebufferResizeEvent() {
  return StateWindow().framebufferresize_event_;
}

EventManager<void(bool)> &IBackendWindow::CommonCursorEnterEvent() {
  return StateWindow().cursorenter_event_;
}

EventManager<void(bool)> &IBackendWindow::CommonFocusEvent() {
  return StateWindow().focus_event_;
}

EventManager<void(double, double)> &IBackendWindow::CommonMouseMoveEvent() {
  return StateWindow().mousemove_event_;
}

EventManager<void(int, int, int, double, double)> &IBackendWindow::CommonMouseButtonEvent() {
  return StateWindow().mousebutton_event_;
}

EventManager<void(double, double)> &IBackendWindow::CommonScrollEvent() {
  return StateWindow().scroll_event_;
}

EventManager<void(int, int, int, int)> &IBackendWindow::CommonKeyEvent() {
  return StateWindow().key_event_;
}

EventManager<void(uint32_t)> &IBackendWindow::CommonCharEvent() {
  return StateWindow().char_event_;
}

EventManager<void(int, const char **)> &IBackendWindow::CommonDropEvent() {
  return StateWindow().drop_event_;
}
}  // namespace grassland::rhi::backend
