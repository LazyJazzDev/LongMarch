#pragma once
#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {

class VulkanCommandContext final : public ICommandContext {
 public:
  VulkanCommandContext(VulkanDevice *device, QueueType type);
  ~VulkanCommandContext() override;
  SubmitResult SubmitNative() override;
  SubmissionStatus WaitForProgress() override;

 private:
  VulkanDevice *device_;
  VkCommandPool pool_{};
  VkFence fence_{};
  bool failed_{};
};

}  // namespace grassland::rhi::backend::vulkan
