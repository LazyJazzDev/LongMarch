#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {

class VulkanWindow;

class VulkanSwapChainImage final : public VulkanImage {
 public:
  VulkanSwapChainImage(ImageDesc desc, VulkanWindow *window, uint64_t generation, VkDevice device);
  ~VulkanSwapChainImage() override;
  bool Acquire();
  const VulkanWindow *Owner() const;
  bool IsValid() const override;
  VkImage NativeImage() const override;
  VkSemaphore AcquireSemaphore() const;
  VkSemaphore RenderSemaphore() const;
  VkResult Present();

 private:
  VulkanWindow *window_;
  uint64_t generation_;
  VkDevice device_;
  VkImage image_{};
  VkSemaphore acquired_{};
  VkSemaphore rendered_{};
  VkFence acquire_fence_{};
  uint32_t index_{};
  bool acquired_image_{};
  std::atomic<bool> presented_{};
};

}  // namespace grassland::rhi::backend::vulkan
