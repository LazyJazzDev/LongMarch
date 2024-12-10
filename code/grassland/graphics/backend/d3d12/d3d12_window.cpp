#include "grassland/graphics/backend/d3d12/d3d12_window.h"

namespace grassland::graphics::backend {
D3D12Window::D3D12Window(int width, int height, const std::string &title)
    : Window(width, height, title) {
}

void D3D12Window::CloseWindow() {
  Window::CloseWindow();
}

}  // namespace grassland::graphics::backend
