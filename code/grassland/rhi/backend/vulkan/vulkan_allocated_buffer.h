#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_buffer.h"
typedef struct VmaAllocation_T *VmaAllocation;
typedef struct VmaAllocator_T *VmaAllocator;

namespace grassland::rhi::backend::vulkan {
class VulkanAllocatedBuffer final : public VulkanBuffer {
 public:
  VulkanAllocatedBuffer(VulkanDevice *device, const BufferDesc &desc);
  ~VulkanAllocatedBuffer() override;
  bool Initialize(bool upload = false);
  VkBuffer NativeBuffer() const override;
  bool Write(const void *data, uint64_t size);

 private:
  VkBuffer buffer_{};
  VmaAllocation allocation_{};
  void *mapped_{};
};
}  // namespace grassland::rhi::backend::vulkan
