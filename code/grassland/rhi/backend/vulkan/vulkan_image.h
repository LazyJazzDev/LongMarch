#pragma once
#include <memory>

#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {

// Common resource access; derived classes define ownership and validity.
class VulkanImage : public Image {
 public:
  class Views {
   public:
    static std::shared_ptr<Views> Create(VkDevice device, VkImage image, ImageFormat format, bool sampled);
    ~Views();
    Views(const Views &) = delete;
    Views &operator=(const Views &) = delete;
    void Reset();
    VkImageView Attachment() const;
    VkImageView Binding() const;

   private:
    explicit Views(VkDevice device);
    VkDevice device_;
    VkImageView attachment_{}, binding_{};
  };

  VkImageView AttachmentView() const;
  VkImageView BindingView() const;
  const void *DeviceIdentity() const override;
  virtual VkImage NativeImage() const = 0;
  VkImageLayout DefaultLayout() const;

 protected:
  VulkanImage(ImageDesc desc, const void *device, VkImageLayout default_layout);

  std::shared_ptr<Views> views_;

 private:
  const void *device_;
  VkImageLayout default_layout_;
};

}  // namespace grassland::rhi::backend::vulkan
