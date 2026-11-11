#include "grassland/rhi/backend/d3d12/d3d12_window.h"

#include "grassland/rhi/backend/d3d12/d3d12_queue.h"
#include "grassland/rhi/backend/d3d12/d3d12_swap_chain_image.h"
#include "grassland/rhi/backend/d3d12/d3d12_window_impl.h"
#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_image_presenter.h"
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/backend_window_imgui.h"
#define GLFW_INCLUDE_NONE
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

namespace grassland::rhi::backend::d3d12 {

namespace {
ImageFormat PresentationFormat(ColorSpace space) {
  switch (space) {
    case COLOR_SPACE_EXTENDED_SRGB_LINEAR:
      return IMAGE_FORMAT_R16G16B16A16_SFLOAT;
    case COLOR_SPACE_HDR10_ST2084:
      return IMAGE_FORMAT_R10G10B10A2_UNORM;
    default:
      return IMAGE_FORMAT_R8G8B8A8_UNORM;
  }
}

DXGI_COLOR_SPACE_TYPE NativeColorSpace(ColorSpace space) {
  switch (space) {
    case COLOR_SPACE_EXTENDED_SRGB_LINEAR:
      return DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709;
    case COLOR_SPACE_HDR10_ST2084:
      return DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
    default:
      return DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
  }
}

// Match the monitor containing most of this window, including before the first swapchain.
bool HDRDisplayEnabled(IDXGIFactory4 *factory, GLFWwindow *window) {
  const auto monitor = MonitorFromWindow(glfwGetWin32Window(window), MONITOR_DEFAULTTONEAREST);
  for (UINT a = 0;; ++a) {
    ComPtr<IDXGIAdapter1> adapter;
    if (factory->EnumAdapters1(a, &adapter) != S_OK)
      break;
    for (UINT o = 0;; ++o) {
      ComPtr<IDXGIOutput> output;
      if (adapter->EnumOutputs(o, &output) != S_OK)
        break;
      ComPtr<IDXGIOutput6> output6;
      DXGI_OUTPUT_DESC1 desc{};
      if (SUCCEEDED(output.As(&output6)) && SUCCEEDED(output6->GetDesc1(&desc)) && desc.Monitor == monitor)
        return desc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
    }
  }
  return false;
}
}  // namespace

D3D12WindowImpl::D3D12WindowImpl(ID3D12Device *device) : device_(device) {
}

D3D12WindowImpl::~D3D12WindowImpl() {
  glfw::UnregisterWindow(GLFWWindow());
  std::lock_guard<std::mutex> lock(queue_->QueueMutex());
  queue_->WaitIdle();
  ReleaseNextImage();

  DestroyWindow();
}

rhi::IWindow *D3D12WindowImpl::Create(ID3D12Device *device, const WindowDesc &desc) {
  auto object = std::unique_ptr<D3D12WindowImpl>(new D3D12WindowImpl(device));
  if (!object->InitializeWindow(device, desc))
    return nullptr;
  if (!object->RebuildSwapChain())
    return nullptr;
  return object.release();
}

bool D3D12WindowImpl::IsFrameValid(uint64_t generation) const {
  return generation_ == generation && valid_;
}

HRESULT D3D12WindowImpl::PresentFrame(uint64_t generation) {
#if !defined(NDEBUG)
  if (!IsFrameValid(generation))
    return E_INVALIDARG;
#endif
  acquired_ = false;
  return swapchain_->Present(mode_ == PRESENT_MODE_VSYNC ? 1 : 0,
                             mode_ == PRESENT_MODE_CONTINUOUS ? DXGI_PRESENT_ALLOW_TEARING : 0);
}

ObjectPtr<IBackendImage> D3D12WindowImpl::AcquireNativeFrame() {
  if (glfwGetWindowAttrib(GLFWWindow(), GLFW_ICONIFIED))
    return {};
  int width, height;
  glfwGetFramebufferSize(GLFWWindow(), &width, &height);
  if (width <= 0 || height <= 0)
    return {};
  // Native presentation can dispatch resize messages before the next PollEvents.
  if (desc_.width != width || desc_.height != height) {
    invalid_ = true;
    return {};
  }
#if !defined(NDEBUG)
  if (!valid_)
    return {};
#endif
#if !defined(NDEBUG)
  if (acquired_) {
    ReportError(-1, "Present or release the previous acquired frame first.");
    return {};
  }
#endif
  const UINT index = swapchain_->GetCurrentBackBufferIndex();
  auto frame = ObjectPtr<IBackendImage>::Adopt(
      ID3D12SwapChainImage::Create(desc_, this, generation_, images_[index].Get(), attachments_[index]));
  acquired_ = true;
  return frame;
}

PresentMode D3D12WindowImpl::GetPresentMode() const {
  return mode_;
}

std::vector<WindowPresentationOption> D3D12WindowImpl::GetSupportedPresentationOptions() const {
  std::vector<WindowPresentationOption> options;
  const bool hdr = HDRDisplayEnabled(device_->Factory(), GLFWWindow());
  for (auto color_space : {COLOR_SPACE_SRGB_NONLINEAR, COLOR_SPACE_EXTENDED_SRGB_LINEAR, COLOR_SPACE_HDR10_ST2084}) {
    if (color_space != COLOR_SPACE_SRGB_NONLINEAR && !hdr)
      continue;
    const auto format = PresentationFormat(color_space);
    options.push_back({PRESENT_MODE_VSYNC, format, color_space});
    if (device_->SupportsTearing())
      options.push_back({PRESENT_MODE_CONTINUOUS, format, color_space});
  }
  return options;
}

bool D3D12WindowImpl::RebuildNativeSwapChain(PresentMode requested_mode) {
  mode_ = requested_mode == PRESENT_MODE_CONTINUOUS && device_->SupportsTearing() ? requested_mode : PRESENT_MODE_VSYNC;
  const auto image_format = PresentationFormat(RequestedColorSpace());
  const auto native_format = NativeImageFormat(image_format);
  valid_ = false;
  if (glfwGetWindowAttrib(GLFWWindow(), GLFW_ICONIFIED))
    return true;
  int width, height;
  glfwGetFramebufferSize(GLFWWindow(), &width, &height);
  if (width <= 0 || height <= 0)
    return true;
  std::lock_guard<std::mutex> lock(queue_->QueueMutex());
  const UINT flags = device_->SupportsTearing() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
  if (swapchain_) {
    if (!queue_->WaitIdle())
      return false;
    ++generation_;
    valid_ = false;
    acquired_ = false;
    for (auto &attachment : attachments_)
      attachment.Reset();
    for (auto &image : images_)
      image.Reset();
    if (!Check(swapchain_->ResizeBuffers(3, width, height, native_format, flags), "ResizeBuffers"))
      return false;
  } else {
    DXGI_SWAP_CHAIN_DESC1 desc{};
    desc.Width = width;
    desc.Height = height;
    desc.Format = native_format;
    desc.SampleDesc.Count = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = 3;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    desc.Flags = flags;
    ComPtr<IDXGISwapChain1> chain;
    const HWND hwnd = glfwGetWin32Window(GLFWWindow());
    if (!Check(device_->Factory()->CreateSwapChainForHwnd(queue_->NativeQueue(), hwnd, &desc, nullptr, nullptr, &chain),
               "CreateSwapChainForHwnd") ||
        !Check(chain.As(&swapchain_), "Query IDXGISwapChain3"))
      return false;
    device_->Factory()->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
  }
  const auto color_space = NativeColorSpace(RequestedColorSpace());
  UINT support{};
  if (!Check(swapchain_->CheckColorSpaceSupport(color_space, &support), "Check swapchain color space") ||
      !(support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT)) {
    ReportError(-1, "Requested DXGI color space cannot be presented.");
    return false;
  }
  if (!Check(swapchain_->SetColorSpace1(color_space), "Set swapchain color space"))
    return false;
  for (UINT i = 0; i < 3; ++i) {
    if (!Check(swapchain_->GetBuffer(i, IID_PPV_ARGS(&images_[i])), "Get swapchain buffer"))
      return false;
    attachments_[i] = ID3D12Image::CreateAttachmentView(device_, images_[i].Get());
    if (!attachments_[i])
      return false;
  }
  desc_ = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), image_format};
  valid_ = true;
  return true;
}

rhi::IWindow *ID3D12Window::Create(ID3D12Device *device, const WindowDesc &desc) {
  return D3D12WindowImpl::Create(device, desc);
}

bool D3D12WindowImpl::EnableImGui(bool enabled, const char *font_file_path, float font_size) {
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

ImGuiContext *D3D12WindowImpl::GetImGuiContext() const {
  return imgui_ ? imgui_->Context() : nullptr;
}

bool D3D12WindowImpl::UpdateImGui(const std::function<void()> &callback) {
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

const void *D3D12WindowImpl::DeviceIdentity() const {
  return device_;
}

bool D3D12WindowImpl::RecordPresentImage(ICommandContext *commands, IImage *image, ColorSpace color_space) {
  ObjPtr<IImage> frame = GetNextImage();
  if (!frame)
    return false;
  if (!presenter_)
    presenter_ = std::make_unique<ImagePresenter>(rhi::QueryInterfacePtr<rhi::IDevice>(device_).Get());
  return presenter_->Record(commands, image, frame.Get(), color_space, GetColorSpace(), GetHDRReferenceWhiteScale());
}

bool D3D12WindowImpl::RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image) {
  return RecordImGui(commands, image, GetColorSpace(), GetHDRReferenceWhiteScale());
}

bool D3D12WindowImpl::RecordImGui(rhi::ICommandContext *commands,
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

rhi::IImage *D3D12WindowImpl::GetNextImage() {
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

IBackendQueue *D3D12WindowImpl::OwnerQueue() const {
  return queue_;
}

bool D3D12WindowImpl::UpdateNextImage() {
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

void D3D12WindowImpl::ReleaseNextImage() {
  std::lock_guard<std::mutex> lock(next_image_mutex_);
  next_image_.Reset();
}

bool D3D12WindowImpl::IsSwapChainInvalid() const {
  return invalid_;
}

bool D3D12WindowImpl::RebuildSwapChain() {
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

bool D3D12WindowImpl::SetPresentMode(PresentMode mode) {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return false;
#endif
  requested_mode_ = mode;
  return RebuildSwapChain();
}

bool D3D12WindowImpl::SetColorSpace(ColorSpace color_space) {
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

ColorSpace D3D12WindowImpl::GetColorSpace() const {
  return color_space_;
}

float D3D12WindowImpl::GetHDRReferenceWhiteScale() const {
#ifdef _WIN32
  float nits;
  return GetHDRReferenceWhiteNits(nits) ? nits / 80.0f : 1.0f;
#else
  // Linux uses fixed source encoding parameters; macOS uses relative EDR.
  return 1.0f;
#endif
}

float D3D12WindowImpl::GetHDRHeadroom() const {
#ifdef __APPLE__
  return CocoaHDRHeadroom(GLFWWindow());
#endif
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

bool D3D12WindowImpl::GetHDRReferenceWhiteNits(float &nits) const {
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

ColorSpace D3D12WindowImpl::RequestedColorSpace() const {
  return requested_color_space_;
}

void D3D12WindowImpl::DestroyWindow() {
  imgui_.reset();
  glfw::UnregisterWindow(window_);
  glfw::DestroyWindow(window_);
}

bool D3D12WindowImpl::InitializeWindow(ID3D12Device *device, const WindowDesc &desc) {
  queue_ = device->GetNativeQueue(QUEUE_TYPE_MAIN);
  requested_mode_ = desc.present_mode;
  requested_color_space_ = desc.color_space;
  window_ = glfw::CreateGLFWWindow(desc.width, desc.height, desc.title.c_str());
  if (!window_)
    return false;
  glfw::RegisterWindow(this, this);
  return true;
}

GLFWwindow *D3D12WindowImpl::GLFWWindow() const {
  return window_;
}

bool D3D12WindowImpl::ShouldClose() const {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return true;
#endif
  return glfwWindowShouldClose(window_);
}

bool D3D12WindowImpl::IsKeyDown(int key) const {
#if !defined(NDEBUG)
  if (!glfw::IsWindowThread())
    return false;
#endif
  return glfwGetKey(window_, key) == GLFW_PRESS;
}

bool D3D12WindowImpl::SetTitle(const std::string &title) {
  return glfw::SetWindowTitle(window_, title.c_str());
}

EventManager<void(int, int)> &D3D12WindowImpl::ResizeEvent() {
  return resize_event_;
}

EventManager<void(int, int)> &D3D12WindowImpl::FramebufferResizeEvent() {
  return framebufferresize_event_;
}

EventManager<void(bool)> &D3D12WindowImpl::CursorEnterEvent() {
  return cursorenter_event_;
}

EventManager<void(bool)> &D3D12WindowImpl::FocusEvent() {
  return focus_event_;
}

EventManager<void(double, double)> &D3D12WindowImpl::MouseMoveEvent() {
  return mousemove_event_;
}

EventManager<void(int, int, int, double, double)> &D3D12WindowImpl::MouseButtonEvent() {
  return mousebutton_event_;
}

EventManager<void(double, double)> &D3D12WindowImpl::ScrollEvent() {
  return scroll_event_;
}

EventManager<void(int, int, int, int)> &D3D12WindowImpl::KeyEvent() {
  return key_event_;
}

EventManager<void(uint32_t)> &D3D12WindowImpl::CharEvent() {
  return char_event_;
}

EventManager<void(int, const char **)> &D3D12WindowImpl::DropEvent() {
  return drop_event_;
}
}  // namespace grassland::rhi::backend::d3d12
