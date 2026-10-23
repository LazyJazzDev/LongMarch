#include "grassland/rhi/backend/d3d12/d3d12_window.h"

#include "grassland/rhi/backend/d3d12/d3d12_swap_chain_image.h"
#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

namespace grassland::rhi::backend::d3d12 {

namespace {
ImageFormat PresentationFormat(ColorSpace space) {
  switch (space) {
    case COLOR_SPACE_EXTENDED_SRGB_LINEAR:
      return IMAGE_FORMAT_RGBA16_FLOAT;
    case COLOR_SPACE_HDR10_ST2084:
      return IMAGE_FORMAT_RGB10A2_UNORM;
    default:
      return IMAGE_FORMAT_RGBA8_UNORM;
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

D3D12Window::D3D12Window(D3D12Device *device) : device_(device) {
}

D3D12Window::~D3D12Window() {
  glfw::UnregisterWindow(GLFWWindow());
  std::lock_guard<std::mutex> lock(device_->QueueMutex());
  device_->WaitIdle();
  ReleaseNextImage();
}

D3D12Window *D3D12Window::Create(D3D12Device *device, const WindowDesc &desc) {
  auto object = std::unique_ptr<D3D12Window>(new D3D12Window(device));
  if (!object->InitializeWindow(desc))
    return nullptr;
  if (!object->RebuildSwapChain())
    return nullptr;
  return object.release();
}

bool D3D12Window::IsFrameValid(uint64_t generation) const {
  return generation_ == generation && valid_;
}

const void *D3D12Window::DeviceIdentity() const {
  return device_;
}

HRESULT D3D12Window::PresentFrame(uint64_t generation) {
#if !defined(NDEBUG)
  if (!IsFrameValid(generation))
    return E_INVALIDARG;
#endif
  acquired_ = false;
  return swapchain_->Present(mode_ == PRESENT_MODE_VSYNC ? 1 : 0,
                             mode_ == PRESENT_MODE_CONTINUOUS ? DXGI_PRESENT_ALLOW_TEARING : 0);
}

ObjectPtr<Image> D3D12Window::AcquireNativeFrame() {
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
  auto frame = ObjectPtr<Image>::Adopt(
      new D3D12SwapChainImage(desc_, this, generation_, images_[index].Get(), attachments_[index]));
  acquired_ = true;
  return frame;
}

PresentMode D3D12Window::GetPresentMode() const {
  return mode_;
}

std::vector<WindowPresentationOption> D3D12Window::GetSupportedPresentationOptions() const {
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

bool D3D12Window::RebuildNativeSwapChain(PresentMode requested_mode) {
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
  std::lock_guard<std::mutex> lock(device_->QueueMutex());
  const UINT flags = device_->SupportsTearing() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
  if (swapchain_) {
    if (!device_->WaitIdle())
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
    if (!Check(
            device_->Factory()->CreateSwapChainForHwnd(device_->NativeQueue(), hwnd, &desc, nullptr, nullptr, &chain),
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
    attachments_[i] = D3D12Image::CreateAttachmentView(device_, images_[i].Get());
    if (!attachments_[i])
      return false;
  }
  desc_ = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), image_format};
  valid_ = true;
  return true;
}

}  // namespace grassland::rhi::backend::d3d12
