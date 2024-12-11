#include "grassland/graphics/backend/d3d12/d3d12_window.h"

namespace grassland::graphics::backend {
D3D12Window::D3D12Window(int width,
                         int height,
                         const std::string &title,
                         D3D12Core *core)
    : Window(width, height, title), core_(core) {
  HWND hwnd = glfwGetWin32Window(GLFWWindow());
  core_->DXGIFactory()->CreateSwapChain(*core_->CommandQueue(), hwnd,
                                        core_->FramesInFlight(), &swap_chain_);
}

void D3D12Window::CloseWindow() {
  swap_chain_.reset();
  Window::CloseWindow();
}

}  // namespace grassland::graphics::backend
