#pragma once
#include "grassland/rhi/backend/backend_cuda_interop.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
#include "grassland/rhi/backend/vulkan/vulkan_buffer.h"

namespace grassland::rhi::backend::vulkan {
class VulkanCUDABuffer final : public VulkanBuffer, public ICUDABuffer {
 public:
  static VulkanCUDABuffer *Create(VulkanDevice *device, const BufferDesc &desc);
  ~VulkanCUDABuffer() override;
  VkBuffer NativeBuffer() const override;
  void *GetDevicePointer() const override;

 private:
  VulkanCUDABuffer(VulkanDevice *device, const BufferDesc &desc);
  VkBuffer buffer_{};
  VkDeviceMemory memory_{};
  CUDAMapping mapping_;
};

class VulkanCUDAInteropTimeline final : public CUDAInteropTimeline {
 public:
  static VulkanCUDAInteropTimeline *Create(VulkanDevice *device);
  ~VulkanCUDAInteropTimeline() override;
  VkSemaphore NativeSemaphore() const;

 private:
  explicit VulkanCUDAInteropTimeline(VulkanDevice *device);
  VkSemaphore semaphore_{};
};
}  // namespace grassland::rhi::backend::vulkan
#endif
