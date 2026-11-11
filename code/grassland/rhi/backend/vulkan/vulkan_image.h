#pragma once
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::IVulkanImage, "bab85247-2604-44fe-b152-1ea15663177a")

namespace grassland::rhi::backend::vulkan {

// Common resource access; derived classes define ownership and validity.
class IVulkanImage : public IBackendImage {
 public:
  virtual VkImageView AttachmentView() const = 0;
  virtual VkImageView BindingView() const = 0;
  const void *DeviceIdentity() const override = 0;
  virtual VkImage NativeImage() const = 0;
  virtual VkImageAspectFlags Aspect() const = 0;
  virtual VkImageLayout DefaultLayout() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan
