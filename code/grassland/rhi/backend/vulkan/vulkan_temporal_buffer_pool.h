#pragma once
#include "grassland/rhi/backend/backend_temporal_buffer_pool.h"
#include "grassland/rhi/backend/vulkan/vulkan_allocated_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanTemporalBufferLease,
                        grassland::rhi::backend::vulkan::IVulkanBuffer,
                        "b8b91e0e-bc72-4ea2-b918-d6135648577f")

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

class VulkanTemporalBufferLeaseImpl final : public Object<rhi::IBuffer, IVulkanTemporalBufferLease> {
 public:
  VulkanTemporalBufferLeaseImpl(VulkanTemporalBufferPool *pool, IVulkanAllocatedBuffer *buffer, size_t level);
  ~VulkanTemporalBufferLeaseImpl() override;
  BufferDesc GetDesc() const override;
  VkBuffer NativeBuffer() const override;
  bool Read(void *data, uint64_t size, uint64_t offset = 0);
  bool Write(const void *data, uint64_t size);

 private:
  VulkanTemporalBufferPool *pool_;
  IVulkanAllocatedBuffer *buffer_;
  size_t level_;

 public:
  IVulkanDevice *Owner() const override {
    return buffer_->Owner();
  }
};

}  // namespace grassland::rhi::backend::vulkan
