#include "grassland/graphics/backend/d3d12/d3d12_window.h"

namespace grassland::graphics::backend {
D3D12Window::D3D12Window(D3D12Core *core,
                         int width,
                         int height,
                         const std::string &title,
                         bool fullscreen,
                         bool resizable,
                         bool enable_hdr)
    : Window(width, height, title, fullscreen, resizable, enable_hdr), core_(core) {
  HWND hwnd = glfwGetWin32Window(GLFWWindow());
  core_->DXGIFactory()->CreateSwapChain(
      *core_->CommandQueue(), hwnd, std::max(std::min(core_->FramesInFlight(), DXGI_MAX_SWAP_CHAIN_BUFFERS), 2),
      enable_hdr_ ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM, &swap_chain_);
  ResizeEvent().RegisterCallback([this](int width, int height) {
    core_->WaitGPU();
    swap_chain_.reset();
    HWND hwnd = glfwGetWin32Window(GLFWWindow());
    core_->DXGIFactory()->CreateSwapChain(
        *core_->CommandQueue(), hwnd, std::max(std::min(core_->FramesInFlight(), DXGI_MAX_SWAP_CHAIN_BUFFERS), 2),
        enable_hdr_ ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM, &swap_chain_);
  });
}

void D3D12Window::CloseWindow() {
  swap_chain_.reset();
  Window::CloseWindow();
  ResizeEvent().UnregisterCallback(swap_chain_recreate_event_id_);
}

d3d12::SwapChain *D3D12Window::SwapChain() const {
  return swap_chain_.get();
}

ID3D12Resource *D3D12Window::CurrentBackBuffer() const {
  return swap_chain_->BackBuffer(swap_chain_->Handle()->GetCurrentBackBufferIndex());
}

}  // namespace grassland::graphics::backend
