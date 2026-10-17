#include "grassland/rhi/backend/vulkan/vulkan_queue.h"

#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {

VulkanQueue::VulkanQueue(VulkanDevice *device, QueueType type) : Queue(device, type) {
}

}  // namespace grassland::rhi::backend::vulkan
