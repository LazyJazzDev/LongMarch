#include "grassland/rhi/backend/vulkan/vulkan_queue.h"

#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"
#include "grassland/rhi/backend/vulkan/vulkan_command_context_impl.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_queue_impl.h"
#include "grassland/rhi/backend/vulkan/vulkan_timeline_impl.h"

namespace grassland::rhi::backend::vulkan {

rhi::ICommandContext *VulkanQueueImpl::CreateCommandContext() {
  return IVulkanCommandContext::Create(this);
}

IVulkanQueue *VulkanQueueImpl::Create(IVulkanDevice *device,
                                      QueueType type,
                                      VkQueue queue,
                                      std::shared_ptr<std::mutex> mutex) {
  auto result = std::unique_ptr<VulkanQueueImpl>(new VulkanQueueImpl(device, type, queue, std::move(mutex)));
  result->timeline_ = IVulkanTimeline::Create(device);
  if (!result->timeline_)
    return nullptr;
  return result.release();
}

VkQueue VulkanQueueImpl::NativeQueue() const {
  return queue_;
}

IVulkanTimeline *VulkanQueueImpl::GetTimeline() const {
  return timeline_.Get();
}

VkResult VulkanQueueImpl::Wait(uint64_t value, uint64_t timeout) const {
  const auto semaphore = timeline_->NativeSemaphore();
  VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  info.semaphoreCount = 1;
  info.pSemaphores = &semaphore;
  info.pValues = &value;
  return vkWaitSemaphores(rhi::QueryInterfacePtr<IVulkanDevice>(Owner()).Get()->NativeDevice(), &info, timeout);
}

VulkanQueueImpl::VulkanQueueImpl(IVulkanDevice *device,
                                 QueueType type,
                                 VkQueue queue,
                                 std::shared_ptr<std::mutex> mutex)
    : queue_(queue) {
  InitializeQueue(device, type, std::move(mutex));
}

}  // namespace grassland::rhi::backend::vulkan

LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::IVulkanQueue, "09666973-0b3a-4165-bb7f-0faf38e0f01a")

namespace grassland::rhi::backend::vulkan {
IVulkanQueue *IVulkanQueue::Create(IVulkanDevice *device,
                                   QueueType type,
                                   VkQueue queue,
                                   std::shared_ptr<std::mutex> mutex) {
  return VulkanQueueImpl::Create(device, type, queue, mutex);
}
}  // namespace grassland::rhi::backend::vulkan
