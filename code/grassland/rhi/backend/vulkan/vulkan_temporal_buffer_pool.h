#pragma once
#include "grassland/rhi/backend/backend_temporal_buffer_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_allocated_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanTemporalBufferLease)

namespace grassland::rhi::backend::vulkan {
using VulkanTemporalBufferPool = TemporalBufferPool<IVulkanDevice, IVulkanAllocatedBuffer, IVulkanTemporalBufferLease>;

class IVulkanTemporalBufferLease : public IVulkanBuffer {
 public:
  static IVulkanTemporalBufferLease *Create(VulkanTemporalBufferPool *pool,
                                            IVulkanAllocatedBuffer *buffer,
                                            size_t level);

  virtual bool Read(void *data, uint64_t size, uint64_t offset = 0) = 0;
  virtual bool Write(const void *data, uint64_t size) = 0;

 protected:
  static void ReturnBuffer(VulkanTemporalBufferPool *pool, IVulkanAllocatedBuffer *buffer, size_t level);
};

}  // namespace grassland::rhi::backend::vulkan
