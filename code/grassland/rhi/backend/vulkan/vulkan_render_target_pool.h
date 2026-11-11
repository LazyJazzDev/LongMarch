#pragma once
#include <map>
#include <mutex>

#include "grassland/rhi/backend/vulkan/vulkan_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_image_views.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"
#include "grassland/rhi/rhi_command_context.h"

namespace grassland::rhi::backend::vulkan {
class VulkanRenderTargetPool final {
  class LeaseImpl;

  struct Entry {
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    VkFramebuffer framebuffer{};
    size_t leases{};
  };

 public:
  class ILease : public IObject {
   public:
    virtual const std::vector<VkImageView> &VulkanImageViews() const = 0;
    virtual VkFramebuffer Framebuffer() const = 0;
  };

  using Lease = ILease;

  VulkanRenderTargetPool(VkDevice device, bool dynamic_rendering);
  ~VulkanRenderTargetPool();
  VkRenderPass RenderPass(const std::vector<ImageFormat> &colors,
                          ImageFormat depth,
                          const RenderingInfo *operations = nullptr);
  ObjectPtr<Lease> Acquire(const std::vector<IVulkanImage *> &images, VkRenderPass pass, VkExtent2D extent);
  void Retire(VkImage image);

 private:
  void ReleaseEntry(Entry *entry);
  void DestroyEntry(Entry &entry);
  VkDevice device_;
  bool dynamic_rendering_;
  std::mutex mutex_;
  std::map<std::vector<int>, VkRenderPass> passes_;
  std::vector<std::unique_ptr<Entry>> entries_;
};

}  // namespace grassland::rhi::backend::vulkan

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::VulkanRenderTargetPool::ILease,
                        grassland::rhi::IObject,
                        "0954731d-c668-4aca-af2c-c3099abc20f4")
