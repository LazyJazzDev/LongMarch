#pragma once
#include <mutex>

#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
// Device-owned render-pass/framebuffer cache. Window retires physical images after GPU idle.
class VulkanRenderTargetPool {
 public:
  explicit VulkanRenderTargetPool(VkDevice device);
  ~VulkanRenderTargetPool();
  VkRenderPass RenderPass(ImageFormat format);
  bool Acquire(VkImage image, const ImageDesc &desc, uint64_t &lease, VkFramebuffer &framebuffer);
  void Release(uint64_t lease);
  void Retire(VkImage image);

 private:
  VkRenderPass CreateRenderPass(ImageFormat format);

  struct Entry {
    VkImage image{};
    VkImageView view{};
    VkFramebuffer framebuffer{};
    uint64_t leases{};
  };

  VkDevice device_;
  std::mutex mutex_;
  VkRenderPass passes_[2]{};
  std::vector<Entry> entries_;
};

VkFormat ColorFormat(ImageFormat format);
}  // namespace grassland::rhi::backend::vulkan
