#pragma once
#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {

class VulkanCommandContext final : public ICommandContext {
 public:
  VulkanCommandContext(std::shared_ptr<VulkanDevice::BackendLifetime> lifetime, QueueType type);
  ~VulkanCommandContext() override;
  SubmitResult SubmitNative() override;
  SubmissionStatus WaitForProgress() override;

 private:
  std::shared_ptr<VulkanDevice::BackendLifetime> lifetime_;
  VkCommandPool pool_{};
  VkFence fence_{};
  bool failed_{};
};

}  // namespace grassland::rhi::backend::vulkan
