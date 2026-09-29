#pragma once
#include <native_window/external_window.h>
#include <vulkan/vulkan.h>

#include "HdrPresentation.h"
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_image.h"

namespace longmarch::harmony {
class Surface {
 public:
  Surface(grassland::graphics::Core *core, OHNativeWindow *window);
  ~Surface();
  void Resize(uint32_t width, uint32_t height, bool hdr);
  bool Present(grassland::graphics::Image *image,
               double zoom = 1,
               double pan_x = 0,
               double pan_y = 0,
               bool linear_demo = false,
               bool encoded_particles = false);

  bool HDR() const {
    return hdr_;
  }

 private:
  void ReleaseSwapchain();

  // Presentation work in flight. The blit waits for the acquired image and
  // signals presentation on the GPU, so the render thread never blocks on it.
  struct Frame {
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE;
    VkSemaphore blitted = VK_NULL_HANDLE;
    VkFence done = VK_NULL_HANDLE;
  };

  static constexpr size_t kFramesInFlight = 2;

  grassland::graphics::backend::VulkanCore *core_;
  OHNativeWindow *window_;
  std::unique_ptr<HdrPresentation> presentation_;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  Frame frames_[kFramesInFlight];
  size_t frame_ = 0;
  std::vector<VkImage> images_;
  VkExtent2D extent_{};
  bool hdr_ = false;
  bool requested_hdr_ = false;
  uint32_t requested_width_ = 0, requested_height_ = 0;
};
}  // namespace longmarch::harmony
