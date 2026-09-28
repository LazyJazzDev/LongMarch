#include "grassland/graphics/backend/vulkan/helper/surface.h"

#include "grassland/graphics/backend/vulkan/helper/instance.h"

namespace grassland::graphics::backend::vulkan {
Surface::Surface(const class Instance *instance, GLFWwindow *window, VkSurfaceKHR surface)
    : instance_(instance),
      window_(window),
      surface_(surface) {
}

Surface::~Surface() {
  vkDestroySurfaceKHR(instance_->Handle(), surface_, nullptr);
}

VkSurfaceKHR Surface::Handle() const {
  return surface_;
}

GLFWwindow *Surface::Window() const {
  return window_;
}

const Instance *Surface::Instance() const {
  return instance_;
}
}  // namespace grassland::graphics::backend::vulkan
