#include "grassland/graphics/backend/d3d12/d3d12_window.h"

namespace grassland::graphics::backend {
D3D12Window::D3D12Window(D3D12Core *core,
                         int width,
                         int height,
                         const std::string &title)
    : Window(width, height, title), core_(core) {
  HWND hwnd = glfwGetWin32Window(GLFWWindow());
  core_->DXGIFactory()->CreateSwapChain(*core_->CommandQueue(), hwnd,
                                        core_->FramesInFlight(), &swap_chain_);
}

void D3D12Window::CloseWindow() {
  swap_chain_.reset();
  Window::CloseWindow();
}

d3d12::SwapChain *D3D12Window::SwapChain() const {
  return swap_chain_.get();
}

ID3D12Resource *D3D12Window::CurrentBackBuffer() const {
  return swap_chain_->BackBuffer(
      swap_chain_->Handle()->GetCurrentBackBufferIndex());
}

}  // namespace grassland::graphics::backend
