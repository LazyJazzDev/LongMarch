#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::vulkan {
class IVulkanAllocatedBuffer;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::vulkan::IVulkanAllocatedBuffer)
#include "grassland/rhi/backend/vulkan/vulkan_buffer.h"
typedef struct VmaAllocation_T *VmaAllocation;
typedef struct VmaAllocator_T *VmaAllocator;

namespace grassland::rhi::backend::vulkan {
class IVulkanAllocatedBuffer : public IVulkanBuffer {
 public:
  static IVulkanAllocatedBuffer *Create(IVulkanDevice *device,
                                        const BufferDesc &desc,
                                        bool upload = false,
                                        bool download = false);
  virtual VkBuffer NativeBuffer() const = 0;
  virtual bool Read(void *data, uint64_t size, uint64_t offset = 0) = 0;
  virtual bool Write(const void *data, uint64_t size) = 0;
};
}  // namespace grassland::rhi::backend::vulkan
