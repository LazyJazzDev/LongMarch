#include "grassland/vulkan/surface.h"

#include "grassland/vulkan/instance.h"

namespace CD::vulkan {
Surface::Surface(const class Instance *instance, GLFWwindow *window, VkSurfaceKHR surface)
    : instance_(instance), window_(window), surface_(surface) {
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
}  // namespace CD::vulkan
