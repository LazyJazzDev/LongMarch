#pragma once
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {

// Common resource access; derived classes define ownership and validity.
class VulkanImage : public Image {
 public:
  const void *DeviceIdentity() const override;
  virtual VkImage NativeImage() const = 0;

 protected:
  VulkanImage(ImageDesc desc, const void *device);

 private:
  const void *device_;
};

}  // namespace grassland::rhi::backend::vulkan
