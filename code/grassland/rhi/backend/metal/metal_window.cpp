#include "grassland/rhi/backend/metal/metal_window.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <list>
#include <mutex>
#include <thread>

#include "grassland/rhi/backend/backend_image_presenter.h"
#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/backend_window_imgui.h"
#include "grassland/rhi/backend/metal/metal_image.h"
#include "grassland/rhi/rhi_device.h"
#include "grassland/rhi/rhi_queue.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace grassland::rhi::backend::metal {
rhi::IWindow *IMetalWindow::Create(IMetalDevice *device, const WindowDesc &desc) {
  return MetalWindowImpl::Create(device, desc);
}

bool MetalWindowImpl::EnableImGui(bool enabled, const char *font_file_path, float font_size) {
  if (!glfw::IsWindowThread() || processing_imgui_)
    return false;
  if (!enabled) {
    imgui_.reset();
    return true;
  }
  if (!imgui_)
    imgui_ =
        WindowImGui::Create(rhi::QueryInterfacePtr<rhi::IDevice>(device_).Get(), window_, font_file_path, font_size);
  return bool(imgui_);
}

ImGuiContext *MetalWindowImpl::GetImGuiContext() const {
  return imgui_ ? imgui_->Context() : nullptr;
}

bool MetalWindowImpl::UpdateImGui(const std::function<void()> &callback) {
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

  return imgui_->Update(callback);
}

const void *MetalWindowImpl::DeviceIdentity() const {
  return device_;
}

bool MetalWindowImpl::RecordPresentImage(ICommandContext *commands, IImage *image, ColorSpace color_space) {
  ObjPtr<IImage> frame = GetNextImage();
  if (!frame)
    return false;
  if (!presenter_)
    presenter_ = std::make_unique<ImagePresenter>(rhi::QueryInterfacePtr<rhi::IDevice>(device_).Get());
  return presenter_->Record(commands, image, frame.Get(), color_space, GetColorSpace(), GetHDRReferenceWhiteScale());
}

bool MetalWindowImpl::RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image) {
  return RecordImGui(commands, image, GetColorSpace(), GetHDRReferenceWhiteScale());
}

bool MetalWindowImpl::RecordImGui(rhi::ICommandContext *commands,
                                  rhi::IImage *image,
                                  ColorSpace color_space,
                                  float white_scale) {
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

  return imgui_->Record(commands, image, color_space, white_scale);
}

rhi::IImage *MetalWindowImpl::GetNextImage() {
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
  return rhi::QueryInterfacePtr<IImage>(next_image_.Get()).Detach();
}

IMetalQueue *MetalWindowImpl::OwnerQueue() const {
  return queue_;
}

bool MetalWindowImpl::UpdateNextImage() {
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

void MetalWindowImpl::ReleaseNextImage() {
  std::lock_guard<std::mutex> lock(next_image_mutex_);
  next_image_.Reset();
}

bool MetalWindowImpl::IsSwapChainInvalid() const {
  return invalid_;
}

bool MetalWindowImpl::RebuildSwapChain() {
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

bool MetalWindowImpl::SetPresentMode(PresentMode mode) {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return false;
#endif
  requested_mode_ = mode;
  return RebuildSwapChain();
}

bool MetalWindowImpl::SetColorSpace(ColorSpace color_space) {
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

ColorSpace MetalWindowImpl::GetColorSpace() const {
  return color_space_;
}

float MetalWindowImpl::GetHDRReferenceWhiteScale() const {
  // macOS uses relative EDR.
  return 1.0f;
}

float MetalWindowImpl::GetHDRHeadroom() const {
  return CocoaHDRHeadroom(GLFWWindow());
}

bool MetalWindowImpl::GetHDRReferenceWhiteNits(float &) const {
  return false;
}

ColorSpace MetalWindowImpl::RequestedColorSpace() const {
  return requested_color_space_;
}

void MetalWindowImpl::DestroyWindow() {
  imgui_.reset();
  presenter_.reset();
  glfw::UnregisterWindow(window_);
  glfw::DestroyWindow(window_);
}

bool MetalWindowImpl::InitializeWindow(IMetalDevice *device, const WindowDesc &desc) {
  queue_ = device->GetNativeQueue(QUEUE_TYPE_MAIN);
  requested_mode_ = desc.present_mode;
  requested_color_space_ = desc.color_space;
  window_ = glfw::CreateGLFWWindow(desc.width, desc.height, desc.title.c_str());
  if (!window_)
    return false;
  glfw::RegisterWindow(this, this);
  return true;
}

GLFWwindow *MetalWindowImpl::GLFWWindow() const {
  return window_;
}

bool MetalWindowImpl::ShouldClose() const {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return true;
#endif
  return glfwWindowShouldClose(window_);
}

bool MetalWindowImpl::IsKeyDown(int key) const {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return false;
#endif
  return glfwGetKey(window_, key) == GLFW_PRESS;
}

bool MetalWindowImpl::SetTitle(const std::string &title) {
  return glfw::SetWindowTitle(window_, title.c_str());
}

EventManager<void(int, int)> &MetalWindowImpl::ResizeEvent() {
  return resize_event_;
}

EventManager<void(int, int)> &MetalWindowImpl::FramebufferResizeEvent() {
  return framebufferresize_event_;
}

EventManager<void(bool)> &MetalWindowImpl::CursorEnterEvent() {
  return cursorenter_event_;
}

EventManager<void(bool)> &MetalWindowImpl::FocusEvent() {
  return focus_event_;
}

EventManager<void(double, double)> &MetalWindowImpl::MouseMoveEvent() {
  return mousemove_event_;
}

EventManager<void(int, int, int, double, double)> &MetalWindowImpl::MouseButtonEvent() {
  return mousebutton_event_;
}

EventManager<void(double, double)> &MetalWindowImpl::ScrollEvent() {
  return scroll_event_;
}

EventManager<void(int, int, int, int)> &MetalWindowImpl::KeyEvent() {
  return key_event_;
}

EventManager<void(uint32_t)> &MetalWindowImpl::CharEvent() {
  return char_event_;
}

EventManager<void(int, const char **)> &MetalWindowImpl::DropEvent() {
  return drop_event_;
}
}  // namespace grassland::rhi::backend::metal
