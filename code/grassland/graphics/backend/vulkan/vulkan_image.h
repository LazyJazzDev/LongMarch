#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"
#include "grassland/graphics/backend/vulkan/vulkan_util.h"

namespace grassland::graphics::backend {

class VulkanImage : public Image {
 public:
  VulkanImage(VulkanCore *core, int width, int height, ImageFormat format);
  ~VulkanImage() override;
  Extent2D Extent() const override;
  ImageFormat Format() const override;
  void UploadData(const void *data) const override;
  void DownloadData(void *data) const override;
  void UploadData(const void *data, const Offset2D &offset, const Extent2D &extent) const override;
  void DownloadData(void *data, const Offset2D &offset, const Extent2D &extent) const override;

  VkImage Handle() const {
    return image_;
  }

  VkImageView ImageView() const {
    return image_view_;
  }

  VkImageAspectFlags Aspect() const {
    return aspect_;
  }

  VkFormat NativeFormat() const {
    return ImageFormatToVkFormat(format_);
  }

 private:
  VulkanCore *core_;
  VkImage image_{VK_NULL_HANDLE};
  VkImageView image_view_{VK_NULL_HANDLE};
  VmaAllocation allocation_{VK_NULL_HANDLE};
  VkExtent2D extent_{};
  VkImageAspectFlags aspect_{};
  ImageFormat format_;
};

}  // namespace grassland::graphics::backend
