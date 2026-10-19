#pragma once
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {

class VulkanQueue final : public Queue {
 public:
  VulkanQueue(VulkanDevice *device, QueueType type);
};

}  // namespace grassland::rhi::backend::vulkan
