#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {

class VulkanWindow;

class VulkanSwapChainImage final : public VulkanImage {
 public:
  VulkanSwapChainImage(ImageDesc desc, VulkanWindow *window, uint64_t generation, VkImage image);
  ~VulkanSwapChainImage() override;
  bool IsValid() const override;
  VkImage NativeImage() const override;
  VulkanWindow *Owner() const;

 private:
  VulkanWindow *window_;
  uint64_t generation_;
  VkImage image_{};
};

}  // namespace grassland::rhi::backend::vulkan
