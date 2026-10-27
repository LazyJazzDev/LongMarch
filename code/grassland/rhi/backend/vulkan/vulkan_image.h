#pragma once
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {

// Common resource access; derived classes define ownership and validity.
class VulkanImage : public Image {
 public:
  class Views {
   public:
    Views() = default;
    Views(Views &&other) noexcept;
    Views &operator=(Views &&other) noexcept;
    bool Initialize(VkDevice device, VkImage image, ImageFormat format, bool sampled);
    ~Views();
    Views(const Views &) = delete;
    Views &operator=(const Views &) = delete;
    void Reset();
    VkImageView Attachment() const;
    VkImageView Binding() const;

   private:
    VkDevice device_{};
    VkImageView attachment_{}, binding_{};
  };

  // Aliasing frame leases share one layout/access state during recording.
  virtual VulkanImage *StateIdentity() {
    return this;
  }

  virtual VkImageView AttachmentView() const = 0;
  virtual VkImageView BindingView() const = 0;
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
