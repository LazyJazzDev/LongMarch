#pragma once
#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {

class VulkanCommandContext final : public CommandContext {
 public:
  VulkanCommandContext(VulkanDevice *device);
  ~VulkanCommandContext() override;
  bool Initialize();
  SubmitResult SubmitNative() override;
  SubmissionStatus WaitForProgress() override;

 private:
  bool RecordClear(Image *image, const std::array<float, 4> &color, bool first_use) override;
  bool RecordPresent(Image *image) override;
  bool EndRecording() override;
  VulkanDevice *device_;
  VkCommandPool pool_{};
  VkCommandBuffer command_{};
  VkFence fence_{};
  bool failed_{};
};

}  // namespace grassland::rhi::backend::vulkan
