#pragma once
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {

class VulkanQueue final : public IQueue {
 public:
  VulkanQueue(VulkanDevice *device, QueueType type);
  std::unique_ptr<ICommandContext> CreateCommandContext() const override;
};

}  // namespace grassland::rhi::backend::vulkan
