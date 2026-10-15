#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {

class VulkanWindow;

class VulkanSwapChainImage final : public VulkanImage {
 public:
  VulkanSwapChainImage(ImageDesc desc, VulkanWindow *window, VkImage image, bool needs_initialization);
  ~VulkanSwapChainImage() override;
  bool IsValid() const override;
  VkImage NativeImage() const override;
  VulkanWindow *Owner() const;
  bool NeedsInitialization() const;
  void MarkInitialized();

 private:
  VulkanWindow *window_;
  VkImage image_{};
  bool needs_initialization_{};
};

}  // namespace grassland::rhi::backend::vulkan
