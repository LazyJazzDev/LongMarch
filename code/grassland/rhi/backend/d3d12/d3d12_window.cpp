#include "grassland/rhi/backend/d3d12/d3d12_window.h"

#include "grassland/rhi/backend/d3d12/d3d12_image.h"
#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

namespace grassland::rhi::backend::d3d12 {

D3D12Window::D3D12Window(D3D12Device *device) : device_(device) {
}

D3D12Window::~D3D12Window() {
  std::lock_guard<std::mutex> lock(device_->QueueMutex());
  device_->WaitIdle();
}

bool D3D12Window::Initialize(const WindowDesc &desc) {
  if (!InitializeWindow(desc))
    return false;
  SetPresentMode(desc.present_mode);
  int width, height;
  glfwGetFramebufferSize(NativeWindow(), &width, &height);
  return Rebuild(width, height);
}

bool D3D12Window::IsFrameValid(uint64_t generation) const {
  return generation_ == generation && valid_;
}

const void *D3D12Window::DeviceIdentity() const {
  return device_;
}

void D3D12Window::CancelFrame(uint64_t generation) {
  if (generation_ == generation)
    acquired_ = false;
}

HRESULT D3D12Window::PresentFrame(uint64_t generation) {
  if (!IsFrameValid(generation))
    return E_INVALIDARG;
  acquired_ = false;
  return swapchain_->Present(mode_ == PresentMode::VSync ? 1 : 0,
                             mode_ == PresentMode::Continuous ? DXGI_PRESENT_ALLOW_TEARING : 0);
}

ObjectPtr<IImage> D3D12Window::AcquireFrame() {
  if (glfwGetWindowAttrib(NativeWindow(), GLFW_ICONIFIED))
    return {};
  int width, height;
  glfwGetFramebufferSize(NativeWindow(), &width, &height);
  if (width <= 0 || height <= 0)
    return {};
  if (!valid_ || desc_.width != width || desc_.height != height) {
    if (!Rebuild(width, height))
      return {};
  }
  if (acquired_) {
    ReportError(-1, "Present or release the previous acquired frame first.");
    return {};
  }
  const UINT index = swapchain_->GetCurrentBackBufferIndex();
  auto rtv = rtv_->GetCPUDescriptorHandleForHeapStart();
  rtv.ptr += index * stride_;
  auto frame = ObjectPtr<IImage>::Adopt(new D3D12Image(desc_, this, generation_, images_[index].Get(), rtv));
  acquired_ = true;
  return frame;
}

bool D3D12Window::SetPresentMode(PresentMode mode) {
  mode_ = mode == PresentMode::Continuous && device_->SupportsTearing() ? mode : PresentMode::VSync;
  return true;
}

PresentMode D3D12Window::GetPresentMode() const {
  return mode_;
}

bool D3D12Window::Rebuild(int width, int height) {
  if (width <= 0 || height <= 0)
    return false;
  std::lock_guard<std::mutex> lock(device_->QueueMutex());
  const UINT flags = device_->SupportsTearing() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
  if (swapchain_) {
    if (!device_->WaitIdle())
      return false;
    ++generation_;
    valid_ = false;
    acquired_ = false;
    for (auto &image : images_)
      image.Reset();
    rtv_.Reset();
    if (!Check(swapchain_->ResizeBuffers(3, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, flags), "ResizeBuffers"))
      return false;
  } else {
    DXGI_SWAP_CHAIN_DESC1 desc{};
    desc.Width = width;
    desc.Height = height;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = 3;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    desc.Flags = flags;
    ComPtr<IDXGISwapChain1> chain;
    const HWND hwnd = glfwGetWin32Window(NativeWindow());
    if (!Check(device_->Factory()->CreateSwapChainForHwnd(device_->Queue(), hwnd, &desc, nullptr, nullptr, &chain),
               "CreateSwapChainForHwnd") ||
        !Check(chain.As(&swapchain_), "Query IDXGISwapChain3"))
      return false;
    device_->Factory()->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
  }
  D3D12_DESCRIPTOR_HEAP_DESC heap{};
  heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
  heap.NumDescriptors = 3;
  if (!Check(device_->Device()->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&rtv_)), "Create RTV heap"))
    return false;
  stride_ = device_->Device()->GetDescriptorHandleIncrementSize(heap.Type);
  auto handle = rtv_->GetCPUDescriptorHandleForHeapStart();
  for (UINT i = 0; i < 3; ++i) {
    if (!Check(swapchain_->GetBuffer(i, IID_PPV_ARGS(&images_[i])), "Get swapchain buffer"))
      return false;
    device_->Device()->CreateRenderTargetView(images_[i].Get(), nullptr, handle);
    handle.ptr += stride_;
  }
  desc_ = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), ImageFormat::RGBA8Unorm};
  valid_ = true;
  return true;
}

}  // namespace grassland::rhi::backend::d3d12
