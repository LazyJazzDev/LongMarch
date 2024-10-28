#pragma once

#include "grassland/vulkan/vulkan_util.h"

namespace grassland::vulkan {
class Surface {
 public:
  Surface(const class Instance *instance,
          GLFWwindow *window,
          VkSurfaceKHR surface);

  ~Surface();

  VkSurfaceKHR Handle() const;

  GLFWwindow *Window() const;

  const class Instance *Instance() const;

 private:
  const class Instance *instance_{};
  GLFWwindow *window_{};
  VkSurfaceKHR surface_{};
};
}  // namespace grassland::vulkan
