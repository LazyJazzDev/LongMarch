#pragma once

#include "grassland/graphics/backend/vulkan/helper/vulkan_util.h"

namespace grassland::graphics::backend::vulkan {
class Surface {
 public:
  Surface(VkInstance instance, GLFWwindow *window, VkSurfaceKHR surface);

  ~Surface();

  VkSurfaceKHR Handle() const;

  GLFWwindow *Window() const;

  VkInstance Instance() const;

 private:
  VkInstance instance_{};
  GLFWwindow *window_{};
  VkSurfaceKHR surface_{};
};
}  // namespace grassland::graphics::backend::vulkan
