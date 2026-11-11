#pragma once
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanImage)

namespace grassland::rhi::backend::vulkan {

// Common resource access; derived classes define ownership and validity.
struct VulkanImageState;

class IVulkanImage : public IBackendImage {
 public:
  virtual VkImageView AttachmentView() const = 0;
  virtual VkImageView BindingView() const = 0;
  const void *DeviceIdentity() const override = 0;
  const void *CommonDeviceIdentity() const;
  virtual VkImage NativeImage() const = 0;
  VkImageLayout DefaultLayout() const;

 protected:
  void InitializeVulkanImage(ImageDesc desc,
                             const void *device,
                             VkImageLayout default_layout,
                             IBackendQueue *queue = nullptr);
  virtual VulkanImageState &StateVulkanImage() = 0;
  virtual const VulkanImageState &StateVulkanImage() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan
