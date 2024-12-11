#include "grassland/graphics/backend/vulkan/vulkan_window.h"

namespace grassland::graphics::backend {

VulkanWindow::VulkanWindow(int width,
                           int height,
                           const std::string &title,
                           VulkanCore *core)
    : Window(width, height, title), core_(core) {
  core_->Instance()->CreateSurfaceFromGLFWWindow(GLFWWindow(), &surface_);
  core_->Device()->CreateSwapchain(surface_.get(), &swap_chain_);
}

void VulkanWindow::CloseWindow() {
  swap_chain_.reset();
  surface_.reset();
  Window::CloseWindow();
}

}  // namespace grassland::graphics::backend
