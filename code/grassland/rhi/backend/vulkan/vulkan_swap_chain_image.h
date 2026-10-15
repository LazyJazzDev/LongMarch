#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {

class VulkanWindow;
class VulkanCommandContext;

class VulkanSwapChainImage final : public VulkanImage {
 public:
  VulkanSwapChainImage(ImageDesc desc, VulkanWindow *window, uint64_t generation);
  ~VulkanSwapChainImage() override;
  bool Acquire();
  bool IsValid() const override;
  VkImage NativeImage() const override;
  VkSemaphore AcquireSemaphore() const;
  void ConsumeAcquireSemaphore(VulkanCommandContext *submission);
  VkSemaphore RenderSemaphore() const;
  VkResult Present();

 private:
  VulkanWindow *window_;
  uint64_t generation_;
  VkImage image_{};
  VkSemaphore acquired_{};
  VkSemaphore rendered_{};
  uint32_t sync_index_{};
  uint32_t index_{};
  bool acquired_image_{};
  bool acquire_wait_pending_{true};
#if !defined(NDEBUG)
  bool present_requested_{};
#endif
};

}  // namespace grassland::rhi::backend::vulkan
