#pragma once
#include <map>
#include <mutex>

#include "grassland/rhi/backend/vulkan/vulkan_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_image_views.h"
#include "grassland/rhi/rhi_command_context.h"

namespace grassland::rhi::backend::vulkan {
class VulkanRenderTargetPool {
  struct Entry {
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    VkFramebuffer framebuffer{};
    size_t leases{};
  };

 public:
  class Lease final : public Object<IObject> {
   public:
    using Object<IObject>::QueryInterface;
    bool QueryInterface(InterfaceID id, void **output) noexcept override;

    Lease(VulkanRenderTargetPool *pool, Entry *entry, const std::vector<VulkanImage *> &images);
    ~Lease() override;
    const std::vector<VkImageView> &VulkanImageViews() const;
    VkFramebuffer Framebuffer() const;

   private:
    VulkanRenderTargetPool *pool_;
    Entry *entry_;
    std::vector<ObjectPtr<VulkanImage>> images_;
  };

  VulkanRenderTargetPool(VkDevice device, bool dynamic_rendering);
  ~VulkanRenderTargetPool();
  VkRenderPass RenderPass(const std::vector<ImageFormat> &colors,
                          ImageFormat depth,
                          const RenderingInfo *operations = nullptr);
  ObjectPtr<Lease> Acquire(const std::vector<VulkanImage *> &images, VkRenderPass pass);
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

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::VulkanRenderTargetPool::Lease)

inline bool grassland::rhi::backend::vulkan::VulkanRenderTargetPool::Lease::QueryInterface(InterfaceID id,
                                                                                           void **output) noexcept {
  if (!output)
    return false;
  if (auto result = MatchInterface<Lease>(id, this)) {
    *output = result;
    AddRef();
    return true;
  }
  return Object<IObject>::QueryInterface(id, output);
}
