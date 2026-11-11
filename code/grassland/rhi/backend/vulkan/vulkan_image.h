#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::vulkan {
class VulkanImage;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::VulkanImage)
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {

// Common resource access; derived classes define ownership and validity.
struct VulkanImageState;

class VulkanImage : public Image {
 public:
  virtual VkImageView AttachmentView() const = 0;
  virtual VkImageView BindingView() const = 0;
  const void *DeviceIdentity() const override;
  virtual VkImage NativeImage() const = 0;
  VkImageLayout DefaultLayout() const;

 protected:
  void InitializeVulkanImage(ImageDesc desc, const void *device, VkImageLayout default_layout, Queue *queue = nullptr);
  virtual VulkanImageState &StateVulkanImage() = 0;
  virtual const VulkanImageState &StateVulkanImage() const = 0;
};

}  // namespace grassland::rhi::backend::vulkan
