#pragma once
#include "grassland/rhi/backend/backend_cuda_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_buffer.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
namespace grassland::rhi::backend::vulkan {
class VulkanCUDABufferImpl final : public Object<rhi::ICUDABuffer, IVulkanBuffer> {
 public:
  static rhi::IBuffer *Create(IVulkanDevice *device, const BufferDesc &desc);
  ~VulkanCUDABufferImpl() override;
  BufferDesc GetDesc() const override;
  void *GetDevicePointer() const override;

  IVulkanDevice *Owner() const override {
    return owner_;
  }

  VkBuffer NativeBuffer() const override;

 private:
  VulkanCUDABufferImpl(IVulkanDevice *device, const BufferDesc &desc, int cuda_device);
  IVulkanDevice *owner_;
  BufferDesc desc_;
  VkBuffer buffer_{};
  VkDeviceMemory memory_{};
  CUDAMapping mapping_;
};
}  // namespace grassland::rhi::backend::vulkan
#endif
