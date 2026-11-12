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

class VulkanAllocatedBufferImpl final : public Object<rhi::IBuffer, IVulkanAllocatedBuffer> {
 public:
  ~VulkanAllocatedBufferImpl() override;
  static IVulkanAllocatedBuffer *Create(IVulkanDevice *device,
                                        const BufferDesc &desc,
                                        bool upload = false,
                                        bool download = false);
  VkBuffer NativeBuffer() const override;
  bool Read(void *data, uint64_t size, uint64_t offset = 0);
  bool Write(const void *data, uint64_t size);

 private:
  VulkanAllocatedBufferImpl(IVulkanDevice *device, const BufferDesc &desc);

  VkBuffer buffer_{};
  VmaAllocation allocation_{};
  void *mapped_{};

  IVulkanDevice *device_;
  BufferDesc desc_;

 public:
  BufferDesc GetDesc() const override {
    return desc_;
  }

  IVulkanDevice *Owner() const override;
};

}  // namespace grassland::rhi::backend::vulkan
