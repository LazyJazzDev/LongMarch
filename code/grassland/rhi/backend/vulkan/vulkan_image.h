#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_util.h"
#include "grassland/rhi/rhi_image.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanImage,
                        grassland::rhi::IObject,
                        "bab85247-2604-44fe-b152-1ea15663177a")

namespace grassland::rhi::backend::vulkan {

// Common resource access; derived classes define ownership and validity.
class IVulkanImage : public IObject {
 public:
  virtual bool SupportsQueue(const IVulkanQueue *queue) const = 0;
  virtual bool SupportsStorage() const = 0;
  virtual const void *DeviceIdentity() const = 0;
  virtual VkImageView AttachmentView() const = 0;
  virtual VkImageView BindingView() const = 0;
  virtual VkImage NativeImage() const = 0;
  virtual VkImageAspectFlags Aspect() const = 0;
  virtual VkImageLayout DefaultLayout() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan
