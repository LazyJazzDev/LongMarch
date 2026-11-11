#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::vulkan {
class IVulkanAllocatedImage;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanAllocatedImage)
#include "grassland/rhi/backend/vulkan/vulkan_image.h"
typedef struct VmaAllocation_T *VmaAllocation;

namespace grassland::rhi::backend::vulkan {
class IVulkanDevice;

class IVulkanAllocatedImage : public VulkanImage {
 public:
  static IVulkanAllocatedImage *Create(IVulkanDevice *device, const ImageDesc &desc);
  virtual bool IsValid() const = 0;
  virtual bool SupportsStorage() const = 0;
  virtual VkImage NativeImage() const = 0;
  virtual VkImageView AttachmentView() const = 0;
  virtual VkImageView BindingView() const = 0;
};

;
}  // namespace grassland::rhi::backend::vulkan
