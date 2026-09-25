#pragma once
#include <native_window/external_window.h>
#include <vulkan/vulkan.h>

#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_image.h"

namespace longmarch::harmony {
class Surface {
 public:
  Surface(grassland::graphics::Core *core, OHNativeWindow *window);
  ~Surface();
  void Resize(uint32_t width, uint32_t height, bool hdr);
  bool Present(grassland::graphics::Image *image, double zoom = 1, double pan_x = 0, double pan_y = 0);

  bool HDR() const {
    return hdr_;
  }

 private:
  void ReleaseSwapchain();
  grassland::graphics::backend::VulkanCore *core_;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  VkFence acquire_fence_ = VK_NULL_HANDLE;
  std::vector<VkImage> images_;
  VkExtent2D extent_{};
  bool hdr_ = false;
  bool requested_hdr_ = false;
  uint32_t requested_width_ = 0, requested_height_ = 0;
};
}  // namespace longmarch::harmony
