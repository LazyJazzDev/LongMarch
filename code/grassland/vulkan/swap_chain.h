#pragma once

#include "grassland/vulkan/device.h"
#include "grassland/vulkan/surface.h"
#include "grassland/vulkan/vulkan_util.h"

namespace grassland::vulkan {

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

class Swapchain {
 public:
  Swapchain(const class Device *device,
            const class Surface *surface,
            VkSwapchainKHR swapchain,
            VkFormat format,
            VkExtent2D extent);

  ~Swapchain();

  VkSwapchainKHR Handle() const {
    return swapchain_;
  }

  const class Device *Device() const {
    return device_;
  }

  const class Surface *Surface() const {
    return surface_;
  }

  VkFormat Format() const {
    return format_;
  }

  VkExtent2D Extent() const {
    return extent_;
  }

  const std::vector<VkImage> &Images() const {
    return images_;
  }

  const std::vector<VkImageView> &ImageViews() const {
    return image_views_;
  }

  VkImage Image(uint32_t index) const {
    return images_[index];
  }

  VkImageView ImageView(uint32_t index) const {
    return image_views_[index];
  }

  uint32_t ImageCount() const {
    return image_count_;
  }

  // Acquire an image from the swap chain
  VkResult AcquireNextImage(uint64_t timeout, VkSemaphore semaphore, VkFence fence, uint32_t *image_index) const;

  static SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);

  static VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats,
                                                    VkFormat preferred_format);

  static VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes);

  static VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities, GLFWwindow *window);

 private:
  void CreateImageViews();

  const class Device *device_;
  const class Surface *surface_;

  VkSwapchainKHR swapchain_{};
  VkFormat format_{};
  VkExtent2D extent_{};
  uint32_t image_count_{};
  std::vector<VkImage> images_;
  std::vector<VkImageView> image_views_;
};
}  // namespace grassland::vulkan
