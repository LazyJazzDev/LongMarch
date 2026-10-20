#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_image.h"
typedef struct VmaAllocation_T *VmaAllocation;

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

class VulkanAllocatedImage final : public VulkanImage {
 public:
  static VulkanAllocatedImage *Create(VulkanDevice *device, const ImageDesc &desc);
  ~VulkanAllocatedImage() override;
  bool IsValid() const override;
  bool SupportsStorage() const override;
  VkImage NativeImage() const override;

 private:
  VulkanAllocatedImage(VulkanDevice *device, ImageDesc desc);
  VulkanDevice *device_;
  VkImage image_{};
  bool storage_{};
  VmaAllocation allocation_{};
};
}  // namespace grassland::rhi::backend::vulkan
