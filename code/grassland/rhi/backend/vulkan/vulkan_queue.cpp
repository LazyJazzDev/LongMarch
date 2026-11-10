#include "grassland/rhi/backend/vulkan/vulkan_queue.h"

#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {

rhi::ICommandContext *VulkanQueue::CreateCommandContext() {
  return VulkanCommandContext::Create(this);
}

VulkanQueue *VulkanQueue::Create(VulkanDevice *device, QueueType type, VkQueue queue) {
  auto result = std::unique_ptr<VulkanQueue>(new VulkanQueue(device, type, queue));
  result->timeline_ = VulkanTimeline::Create(device);
  if (!result->timeline_)
    return nullptr;
  return result.release();
}

VkQueue VulkanQueue::NativeQueue() const {
  return queue_;
}

VulkanTimeline *VulkanQueue::GetTimeline() const {
  return timeline_.get();
}

VkResult VulkanQueue::Wait(uint64_t value, uint64_t timeout) const {
  const auto semaphore = timeline_->NativeSemaphore();
  VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  info.semaphoreCount = 1;
  info.pSemaphores = &semaphore;
  info.pValues = &value;
  return vkWaitSemaphores(static_cast<VulkanDevice *>(Owner())->NativeDevice(), &info, timeout);
}

VulkanQueue::VulkanQueue(VulkanDevice *device, QueueType type, VkQueue queue) : Queue(device, type), queue_(queue) {
}

}  // namespace grassland::rhi::backend::vulkan
