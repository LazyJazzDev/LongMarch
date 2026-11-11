#include "grassland/rhi/backend/vulkan/vulkan_timeline.h"

#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_timeline_impl.h"

namespace grassland::rhi::backend::vulkan {

VulkanTimelineImpl::VulkanTimelineImpl(IVulkanDevice *device) : device_(device->NativeDevice()) {
  InitializeTimeline(device);
}

ObjectPtr<IVulkanTimeline> VulkanTimelineImpl::Create(IVulkanDevice *device) {
  auto result = ObjectPtr<VulkanTimelineImpl>::Adopt(new VulkanTimelineImpl(device));
  VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  info.pNext = &type;
  if (!Check(vkCreateSemaphore(result->device_, &info, nullptr, &result->semaphore_), "Create queue timeline"))
    return {};
  return result;
}

VulkanTimelineImpl::~VulkanTimelineImpl() {
  if (semaphore_)
    vkDestroySemaphore(device_, semaphore_, nullptr);
}

bool VulkanTimelineImpl::GetCompletedValue(uint64_t &value) const {
  return Check(vkGetSemaphoreCounterValue(device_, semaphore_, &value), "Query timeline completion");
}

bool VulkanTimelineImpl::WaitHost(uint64_t value) const {
  VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  info.semaphoreCount = 1;
  info.pSemaphores = &semaphore_;
  info.pValues = &value;
  return Check(vkWaitSemaphores(device_, &info, UINT64_MAX), "Wait for timeline");
}

VkSemaphore VulkanTimelineImpl::NativeSemaphore() const {
  return semaphore_;
}

ObjectPtr<IVulkanTimeline> IVulkanTimeline::Create(IVulkanDevice *device) {
  return VulkanTimelineImpl::Create(device);
}
}  // namespace grassland::rhi::backend::vulkan
