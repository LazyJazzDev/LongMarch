#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"
typedef struct VmaAllocation_T *VmaAllocation;

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanAllocatedImage)

namespace grassland::rhi::backend::vulkan {
class IVulkanAllocatedImage : public IVulkanImage {
 public:
  static IVulkanAllocatedImage *Create(IVulkanDevice *device, const ImageDesc &desc);
  virtual bool IsValid() const = 0;
  virtual bool SupportsStorage() const = 0;
  virtual VkImage NativeImage() const = 0;
  virtual VkImageView AttachmentView() const = 0;
  virtual VkImageView BindingView() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan
