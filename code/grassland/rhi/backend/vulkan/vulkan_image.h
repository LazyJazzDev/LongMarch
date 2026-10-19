#pragma once
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {

// Common resource access; derived classes define ownership and validity.
class VulkanImage : public Image {
 public:
  const void *DeviceIdentity() const override;
  virtual VkImage NativeImage() const = 0;
  VkImageLayout DefaultLayout() const;

 protected:
  VulkanImage(ImageDesc desc, const void *device, VkImageLayout default_layout);

 private:
  const void *device_;
  VkImageLayout default_layout_;
};

}  // namespace grassland::rhi::backend::vulkan
