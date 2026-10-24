#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_buffer.h"
typedef struct VmaAllocation_T *VmaAllocation;
typedef struct VmaAllocator_T *VmaAllocator;

namespace grassland::rhi::backend::vulkan {
class VulkanAllocatedBuffer final : public VulkanBuffer {
 public:
  ~VulkanAllocatedBuffer() override;
  static VulkanAllocatedBuffer *Create(VulkanDevice *device,
                                       const BufferDesc &desc,
                                       bool upload = false,
                                       bool download = false);
  VkBuffer NativeBuffer() const override;
  bool Read(void *data, uint64_t size, uint64_t offset = 0);
  bool Write(const void *data, uint64_t size);

 private:
  VulkanAllocatedBuffer(VulkanDevice *device, const BufferDesc &desc);

  VkBuffer buffer_{};
  VmaAllocation allocation_{};
  void *mapped_{};
};
}  // namespace grassland::rhi::backend::vulkan
