#include "grassland/graphics/backend/vulkan/helper/surface.h"

namespace grassland::graphics::backend::vulkan {
Surface::Surface(VkInstance instance, GLFWwindow *window, VkSurfaceKHR surface)
    : instance_(instance),
      window_(window),
      surface_(surface) {
}

Surface::~Surface() {
  vkDestroySurfaceKHR(instance_, surface_, nullptr);
}

VkSurfaceKHR Surface::Handle() const {
  return surface_;
}

GLFWwindow *Surface::Window() const {
  return window_;
}

VkInstance Surface::Instance() const {
  return instance_;
}
}  // namespace grassland::graphics::backend::vulkan
