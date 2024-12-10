#include "grassland/graphics/backend/vulkan/vulkan_window.h"

namespace grassland::graphics::backend {

VulkanWindow::VulkanWindow(int width, int height, const std::string &title)
    : Window(width, height, title) {
}

void VulkanWindow::CloseWindow() {
  Window::CloseWindow();
}

}  // namespace grassland::graphics::backend
