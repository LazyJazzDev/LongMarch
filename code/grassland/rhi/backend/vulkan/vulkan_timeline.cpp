#include "grassland/rhi/backend/vulkan/vulkan_timeline.h"

#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
VulkanTimeline::VulkanTimeline(VulkanDevice *device) : backend::Timeline(device) {
}

VulkanTimeline *VulkanTimeline::Create(VulkanDevice *device) {
  auto result = std::unique_ptr<VulkanTimeline>(new VulkanTimeline(device));
  VkSemaphoreTypeCreateInfo timeline{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  timeline.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo create{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  create.pNext = &timeline;
  if (!Check(vkCreateSemaphore(device->NativeDevice(), &create, nullptr, &result->semaphore_), "Create queue timeline"))
    return nullptr;
  return result.release();
}

VulkanTimeline::~VulkanTimeline() {
  if (semaphore_)
    vkDestroySemaphore(static_cast<VulkanDevice *>(Owner())->NativeDevice(), semaphore_, nullptr);
}

VkSemaphore VulkanTimeline::Native() const {
  return semaphore_;
}

bool VulkanTimeline::GetCompletedValue(uint64_t &value) const {
  return vkGetSemaphoreCounterValue(static_cast<VulkanDevice *>(Owner())->NativeDevice(), semaphore_, &value) ==
         VK_SUCCESS;
}

VkResult VulkanTimeline::Wait(uint64_t value, uint64_t timeout) const {
  VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wait.semaphoreCount = 1;
  wait.pSemaphores = &semaphore_;
  wait.pValues = &value;
  return vkWaitSemaphores(static_cast<VulkanDevice *>(Owner())->NativeDevice(), &wait, timeout);
}

}  // namespace grassland::rhi::backend::vulkan
