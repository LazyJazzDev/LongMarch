#pragma once
#include <vector>

#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
class VulkanImage;
class VulkanWindow;
class VulkanSwapChainImage;

class VulkanCommandContext final : public CommandContext {
 public:
  VulkanCommandContext(VulkanDevice *device);
  ~VulkanCommandContext() override;
  bool Initialize();
  bool CmdClearColor(rhi::IImage *image, const std::array<float, 4> &color) override;
  bool CmdPresent(rhi::IWindow *window) override;
  SubmitResult SubmitNative() override;
  SubmissionStatus WaitForProgress() override;

 private:
  bool EndRecording() override;
  bool TransitionImage(VulkanImage *image, VkImageLayout layout, VkPipelineStageFlags stages, VkAccessFlags access);

  struct ImageState {
    VkImageLayout layout;
    VkPipelineStageFlags stages;
    VkAccessFlags access;
  };

  std::unordered_map<VulkanImage *, ImageState> image_states_;
  std::vector<std::pair<VulkanWindow *, VulkanSwapChainImage *>> presents_;
  VulkanDevice *device_;
  VkCommandPool pool_{};
  VkCommandBuffer command_{};
  VkFence fence_{};
  bool submission_failed_{};
};

}  // namespace grassland::rhi::backend::vulkan
