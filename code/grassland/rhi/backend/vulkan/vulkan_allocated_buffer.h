#pragma once
#include "grassland/rhi/backend/vulkan/vulkan_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"
typedef struct VmaAllocation_T *VmaAllocation;
typedef struct VmaAllocator_T *VmaAllocator;

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanAllocatedBuffer,
                        grassland::rhi::backend::vulkan::IVulkanBuffer,
                        "0e429c4d-302f-421e-90d8-b29a6593b7d3")

namespace grassland::rhi::backend::vulkan {
class IVulkanAllocatedBuffer : public IVulkanBuffer {
 public:
  static IVulkanAllocatedBuffer *Create(IVulkanDevice *device,
                                        const BufferDesc &desc,
                                        bool upload = false,
                                        bool download = false);

  virtual bool Read(void *data, uint64_t size, uint64_t offset = 0) = 0;
  virtual bool Write(const void *data, uint64_t size) = 0;
};
}  // namespace grassland::rhi::backend::vulkan
