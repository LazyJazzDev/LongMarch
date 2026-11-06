#include "grassland/rhi/backend/vulkan/vulkan_queue.h"

#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VulkanQueue::VulkanQueue(VulkanDevice *device,
                         QueueType type,
                         VkQueue queue,
                         VulkanQueue *physical_owner,
                         ObjectPtr<VulkanTimeline> timeline)
    : Queue(device, type, std::move(timeline)),
      queue_(queue),
      physical_owner_(physical_owner) {
}

VulkanQueue *VulkanQueue::Create(VulkanDevice *device, QueueType type, VkQueue queue, VulkanQueue *physical_owner) {
  auto timeline = ObjectPtr<VulkanTimeline>::Adopt(VulkanTimeline::Create(device));
  if (!timeline)
    return nullptr;
  auto result = std::unique_ptr<VulkanQueue>(new VulkanQueue(device, type, queue, physical_owner, std::move(timeline)));
  return result.release();
}

VulkanQueue::~VulkanQueue() = default;

VkQueue VulkanQueue::Native() const {
  return queue_;
}

VkSemaphore VulkanQueue::Timeline() const {
  return static_cast<VulkanTimeline *>(TimelineState())->Native();
}

const void *VulkanQueue::NativeIdentity() const {
  return queue_;
}

std::mutex &VulkanQueue::Mutex() {
  return physical_owner_ ? physical_owner_->Mutex() : Queue::Mutex();
}

VkResult VulkanQueue::Wait(uint64_t value, uint64_t timeout) const {
  return static_cast<VulkanTimeline *>(TimelineState())->Wait(value, timeout);
}

ICommandContext *VulkanQueue::CreateCommandContext() {
  return VulkanCommandContext::Create(this);
}

}  // namespace grassland::rhi::backend::vulkan
