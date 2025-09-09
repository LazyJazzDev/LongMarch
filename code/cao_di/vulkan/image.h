#pragma once

#include "cao_di/vulkan/device.h"

namespace CD::vulkan {
class Image {
 public:
  Image(const class Device *device,
        VkFormat format,
        VkExtent2D extent,
        VkImageUsageFlags usage,
        VkImageAspectFlags aspect,
        VkSampleCountFlagBits sample_count,
        VkImage image,
        VkImageView image_view,
        VmaAllocation allocation);
  ~Image();

  const class Device *Device() const {
    return device_;
  }

  VkImage Handle() const {
    return image_;
  }

  VkImageView ImageView() const {
    return image_view_;
  }

  VkFormat Format() const {
    return format_;
  }

  VkExtent2D Extent() const {
    return extent_;
  }

  uint32_t Width() const {
    return extent_.width;
  }

  uint32_t Height() const {
    return extent_.height;
  }

  VkImageUsageFlags Usage() const {
    return usage_;
  }

  VkImageAspectFlags Aspect() const {
    return aspect_;
  }

  VkSampleCountFlagBits SampleCount() const {
    return sample_count_;
  }

  void ClearColor(VkCommandBuffer command_buffer, VkClearColorValue clear_color);

  VkResult Resize(VkExtent2D extent);

  void FetchPixelData(CommandPool *command_pool,
                      Queue *queue,
                      VkRect2D rect,
                      void *data,
                      VkDeviceSize size,
                      VkImageLayout image_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) const;

 private:
  const class Device *device_{};
  VkFormat format_{};
  VkExtent2D extent_{};
  VkImageUsageFlags usage_{};
  VkImageAspectFlags aspect_{};
  VkSampleCountFlagBits sample_count_{};
  VkImage image_{};
  VkImageView image_view_{};
  VmaAllocation allocation_{};
};

void TransitImageLayout(VkCommandBuffer command_buffer,
                        VkImage image,
                        VkImageLayout old_layout,
                        VkImageLayout new_layout,
                        VkPipelineStageFlags src_stage_flags,
                        VkPipelineStageFlags dst_stage_flags,
                        VkAccessFlags src_access_flags,
                        VkAccessFlags dst_access_flags,
                        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

void UploadImage(Queue *queue, CommandPool *command_pool, Image *image, const void *data, VkDeviceSize size);

void BlitImage(VkCommandBuffer cmd_buffer,
               Image *src_image,
               Image *dst_image,
               int num_region = 0,
               VkImageBlit *regions = nullptr,
               VkFilter filter = VK_FILTER_LINEAR);

}  // namespace CD::vulkan
