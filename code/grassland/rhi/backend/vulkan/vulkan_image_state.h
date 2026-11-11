#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/vulkan/vulkan_image.h"

namespace grassland::rhi::backend::vulkan {
struct VulkanImageState final {
  VulkanImageState();
  ~VulkanImageState();
  const void *device_{};
  VkImageLayout default_layout_{};
};
}  // namespace grassland::rhi::backend::vulkan
