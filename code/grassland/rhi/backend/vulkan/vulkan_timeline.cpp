#include "grassland/rhi/backend/vulkan/vulkan_timeline.h"

#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {

VulkanTimeline::VulkanTimeline(VulkanDevice *device) : Timeline(device), device_(device->NativeDevice()) {
}

std::unique_ptr<VulkanTimeline> VulkanTimeline::Create(VulkanDevice *device) {
  auto result = std::unique_ptr<VulkanTimeline>(new VulkanTimeline(device));
  VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  info.pNext = &type;
  if (!Check(vkCreateSemaphore(result->device_, &info, nullptr, &result->semaphore_), "Create queue timeline"))
    return {};
  return result;
}

VulkanTimeline::~VulkanTimeline() {
  if (semaphore_)
    vkDestroySemaphore(device_, semaphore_, nullptr);
}

bool VulkanTimeline::GetCompletedValue(uint64_t &value) const {
  return Check(vkGetSemaphoreCounterValue(device_, semaphore_, &value), "Query timeline completion");
}

bool VulkanTimeline::WaitHost(uint64_t value) const {
  VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  info.semaphoreCount = 1;
  info.pSemaphores = &semaphore_;
  info.pValues = &value;
  return Check(vkWaitSemaphores(device_, &info, UINT64_MAX), "Wait for timeline");
}

VkSemaphore VulkanTimeline::NativeSemaphore() const {
  return semaphore_;
}

}  // namespace grassland::rhi::backend::vulkan
