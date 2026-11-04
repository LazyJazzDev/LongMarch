#pragma once
#include "grassland/rhi/backend/backend_cuda_buffer.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
#include "grassland/rhi/backend/vulkan/vulkan_util.h"

namespace grassland::rhi::backend::vulkan {
class VulkanDevice;

// Implements the most-derived interface directly: one interface chain and reference count.
class VulkanCUDABuffer final : public Object<ICUDABuffer> {
 public:
  static VulkanCUDABuffer *Create(VulkanDevice *device, const BufferDesc &desc);
  BufferDesc GetDesc() const override;
  void *GetDevicePointer() const override;
  ~VulkanCUDABuffer() override;

 private:
  VulkanCUDABuffer(VulkanDevice *device, const BufferDesc &desc, int cuda_device);
  BufferDesc desc_;
  VulkanDevice *device_;
  VkBuffer buffer_{};
  VkDeviceMemory memory_{};
  CUDAMapping mapping_;
};
}  // namespace grassland::rhi::backend::vulkan
#endif
