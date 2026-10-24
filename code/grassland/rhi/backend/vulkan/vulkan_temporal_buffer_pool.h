#pragma once
#include "grassland/rhi/backend/backend_temporal_buffer_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_allocated_buffer.h"

namespace grassland::rhi::backend::vulkan {
class VulkanTemporalBufferLease;
using VulkanTemporalBufferPool = TemporalBufferPool<VulkanDevice, VulkanAllocatedBuffer, VulkanTemporalBufferLease>;

class VulkanTemporalBufferLease final : public VulkanBuffer {
 public:
  VulkanTemporalBufferLease(VulkanTemporalBufferPool *pool, VulkanAllocatedBuffer *buffer, size_t level);
  ~VulkanTemporalBufferLease() override;
  BufferDesc GetDesc() const override;
  VkBuffer NativeBuffer() const override;
  bool Read(void *data, uint64_t size, uint64_t offset = 0);
  bool Write(const void *data, uint64_t size);

 private:
  VulkanTemporalBufferPool *pool_;
  VulkanAllocatedBuffer *buffer_;
  size_t level_;
};

}  // namespace grassland::rhi::backend::vulkan
