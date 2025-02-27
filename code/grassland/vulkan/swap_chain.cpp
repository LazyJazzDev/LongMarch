#include "grassland/vulkan/swap_chain.h"

namespace grassland::vulkan {
Swapchain::Swapchain(const class grassland::vulkan::Device *device,
                     const class grassland::vulkan::Surface *surface,
                     VkSwapchainKHR swapchain,
                     VkFormat format,
                     VkExtent2D extent)
    : device_(device), surface_(surface), swapchain_(swapchain), format_(format), extent_(extent) {
  CreateImageViews();
}

Swapchain::~Swapchain() {
  for (auto imageView : image_views_) {
    vkDestroyImageView(device_->Handle(), imageView, nullptr);
  }
  vkDestroySwapchainKHR(device_->Handle(), swapchain_, nullptr);
}

void Swapchain::CreateImageViews() {
  vkGetSwapchainImagesKHR(device_->Handle(), swapchain_, &image_count_, nullptr);
  images_.resize(image_count_);
  vkGetSwapchainImagesKHR(device_->Handle(), swapchain_, &image_count_, images_.data());
  image_views_.resize(image_count_);

  for (size_t i = 0; i < image_count_; i++) {
    VkImageViewCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = images_[i];
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = format_;
    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device_->Handle(), &createInfo, nullptr, &image_views_[i]) != VK_SUCCESS) {
      ThrowError("failed to create image views!");
    }
  }
}

SwapChainSupportDetails Swapchain::QuerySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface) {
  SwapChainSupportDetails details;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
  uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

  if (formatCount != 0) {
    details.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
  }
  uint32_t presentModeCount;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

  if (presentModeCount != 0) {
    details.presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
  }
  return details;
}

VkSurfaceFormatKHR Swapchain::ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats,
                                                      VkFormat preferred_format) {
  VkSurfaceFormatKHR result = availableFormats[0];

  // for (const auto &availableFormat : availableFormats) {
  //   if (availableFormat.format == preferred_format) {
  //     result = availableFormat;
  //   } else if (availableFormat.format == VK_FORMAT_R8G8B8A8_UNORM &&
  //              availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
  //     result = availableFormat;
  //   }
  // }

  result.format = preferred_format;
  result.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  if (result.format == VK_FORMAT_R16G16B16A16_SFLOAT) {
    result.colorSpace = VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT;
  }

  return result;
}

VkPresentModeKHR Swapchain::ChooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes) {
  for (const auto &availablePresentMode : availablePresentModes) {
    if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return availablePresentMode;
    }
  }
  // https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Swap_chain
  // Only the VK_PRESENT_MODE_FIFO_KHR mode is guaranteed to be available
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D Swapchain::ChooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities, GLFWwindow *window) {
  if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  } else {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

    actualExtent.width =
        std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    actualExtent.height =
        std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

    return actualExtent;
  }
}

VkResult Swapchain::AcquireNextImage(uint64_t timeout,
                                     VkSemaphore semaphore,
                                     VkFence fence,
                                     uint32_t *image_index) const {
  return vkAcquireNextImageKHR(device_->Handle(), swapchain_, timeout, semaphore, fence, image_index);
}
}  // namespace grassland::vulkan
