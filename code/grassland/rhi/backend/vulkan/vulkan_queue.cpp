#include "grassland/rhi/backend/vulkan/vulkan_queue.h"

#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VulkanQueue::VulkanQueue(VulkanDevice *device, QueueType type, VkQueue queue, VulkanQueue *physical_owner)
    : Queue(device, type),
      queue_(queue),
      physical_owner_(physical_owner) {
}

VulkanQueue *VulkanQueue::Create(VulkanDevice *device, QueueType type, VkQueue queue, VulkanQueue *physical_owner) {
  auto result = std::unique_ptr<VulkanQueue>(new VulkanQueue(device, type, queue, physical_owner));
  VkSemaphoreTypeCreateInfo timeline{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  timeline.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo create{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  create.pNext = &timeline;
  if (!Check(vkCreateSemaphore(device->NativeDevice(), &create, nullptr, &result->timeline_), "Create queue timeline"))
    return nullptr;
  return result.release();
}

VulkanQueue::~VulkanQueue() {
  if (timeline_)
    vkDestroySemaphore(static_cast<VulkanDevice *>(Owner())->NativeDevice(), timeline_, nullptr);
}

VkQueue VulkanQueue::Native() const {
  return queue_;
}

VkSemaphore VulkanQueue::Timeline() const {
  return timeline_;
}

const void *VulkanQueue::NativeIdentity() const {
  return queue_;
}

std::mutex &VulkanQueue::Mutex() {
  return physical_owner_ ? physical_owner_->Mutex() : Queue::Mutex();
}

bool VulkanQueue::QueryTimeline(uint64_t &value) const {
  return vkGetSemaphoreCounterValue(static_cast<VulkanDevice *>(Owner())->NativeDevice(), timeline_, &value) ==
         VK_SUCCESS;
}

VkResult VulkanQueue::Wait(uint64_t value, uint64_t timeout) const {
  VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wait.semaphoreCount = 1;
  wait.pSemaphores = &timeline_;
  wait.pValues = &value;
  return vkWaitSemaphores(static_cast<VulkanDevice *>(Owner())->NativeDevice(), &wait, timeout);
}

ICommandContext *VulkanQueue::CreateCommandContext() {
  return VulkanCommandContext::Create(this);
}

}  // namespace grassland::rhi::backend::vulkan
