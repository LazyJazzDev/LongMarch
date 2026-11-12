#pragma once
#include "grassland/rhi/backend/backend_cuda_timeline.h"
#include "grassland/rhi/backend/vulkan/vulkan_timeline.h"
#include "grassland/rhi/rhi_cuda_timeline.h"
#if defined(LONGMARCH_CUDA_RUNTIME)
namespace grassland::rhi::backend::vulkan {
ObjectPtr<IVulkanTimeline> CreateVulkanCUDATimeline(IVulkanDevice *device);

class VulkanCUDATimelineImpl final : public Object<ICUDATimeline, IVulkanTimeline, IBackendCUDATimeline> {
 public:
  static ObjectPtr<IVulkanTimeline> Create(IVulkanDevice *device);
  ~VulkanCUDATimelineImpl() override;
  bool GetCompletedValue(uint64_t &value) const override;
  bool WaitHost(uint64_t value) const override;
  VkSemaphore NativeSemaphore() const override;

  IVulkanDevice *Owner() const override {
    return owner_;
  }

  const void *CUDAOwner() const override {
    return owner_;
  }

  cudaExternalSemaphore_t CUDASemaphore() const override {
    return cuda_.Get();
  }

  bool Wait(cudaStream_t stream, const std::vector<Event> &events) override {
    return cuda_.Wait(owner_, stream, events);
  }

  Event Signal(cudaStream_t stream, uint64_t value) override {
    return cuda_.Signal(stream, value) ? Event{this, value} : Event{};
  }

 private:
  explicit VulkanCUDATimelineImpl(IVulkanDevice *device);
  IVulkanDevice *owner_;
  VkSemaphore semaphore_{};
  CUDAExternalSemaphore cuda_;
};

}  // namespace grassland::rhi::backend::vulkan
#endif
