#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"

#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/vulkan/vulkan_image.h"

namespace grassland::rhi::backend::vulkan {

VulkanCommandContext::VulkanCommandContext(VulkanDevice *device, QueueType type)
    : CommandContext(device, type),
      device_(device) {
}

VulkanCommandContext::~VulkanCommandContext() {
  if (fence_)
    vkDestroyFence(device_->NativeDevice(), fence_, nullptr);
  if (pool_)
    vkDestroyCommandPool(device_->NativeDevice(), pool_, nullptr);
}

CommandContext::SubmitResult VulkanCommandContext::SubmitNative() {
  if (Frame() && !glfw::IsWindowThread()) {
    ReportError(-1, "Present must run on the window thread.");
    return SubmitResult::Rejected;
  }
  const auto device = device_->NativeDevice();
  VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool.queueFamilyIndex = device_->QueueFamily();
  pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (!Check(vkCreateCommandPool(device, &pool, nullptr, &pool_), "Create command pool") ||
      !Check(vkCreateFence(device, &fence, nullptr, &fence_), "Create submission fence"))
    return SubmitResult::Rejected;
  VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocate.commandPool = pool_;
  allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate.commandBufferCount = 1;
  VkCommandBuffer command{};
  if (!Check(vkAllocateCommandBuffers(device, &allocate, &command), "Allocate command buffer"))
    return SubmitResult::Rejected;
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (!Check(vkBeginCommandBuffer(command, &begin), "Begin command buffer"))
    return SubmitResult::Rejected;
  auto frame = static_cast<VulkanImage *>(Frame().Get());
  VkSemaphore acquired{}, rendered{};
  if (frame) {
    acquired = frame->AcquireSemaphore();
    rendered = frame->RenderSemaphore();
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = frame->NativeImage();
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
    VkClearColorValue color{};
    std::copy(Color().begin(), Color().end(), color.float32);
    vkCmdClearColorImage(command, barrier.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1,
                         &barrier.subresourceRange);
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = 0;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &barrier);
  }
  if (!Check(vkEndCommandBuffer(command), "End command buffer"))
    return SubmitResult::Rejected;
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &command;
  if (frame) {
    submit.waitSemaphoreCount = submit.signalSemaphoreCount = 1;
    submit.pWaitSemaphores = &acquired;
    submit.pWaitDstStageMask = &stage;
    submit.pSignalSemaphores = &rendered;
  }
  std::lock_guard<std::mutex> lock(device_->QueueMutex());
  if (!Check(vkQueueSubmit(device_->NativeQueue(), 1, &submit, fence_), "vkQueueSubmit"))
    return SubmitResult::Rejected;
  MarkSubmitted();
  failed_ = true;
  failed_ = frame && !Check(frame->Present(), "vkQueuePresentKHR");
  return failed_ ? SubmitResult::FailedInFlight : SubmitResult::Submitted;
}

SubmissionStatus VulkanCommandContext::WaitForProgress() {
  const auto result = vkWaitForFences(device_->NativeDevice(), 1, &fence_, VK_TRUE, 50000000);
  if (result == VK_TIMEOUT)
    return SubmissionStatus::Pending;
  if (result == VK_ERROR_DEVICE_LOST)
    return SubmissionStatus::Failed;
  if (result != VK_SUCCESS)
    failed_ = true;
  if (failed_) {
    std::lock_guard<std::mutex> lock(device_->QueueMutex());
    const auto idle = vkDeviceWaitIdle(device_->NativeDevice());
    return idle == VK_SUCCESS || idle == VK_ERROR_DEVICE_LOST ? SubmissionStatus::Failed : SubmissionStatus::Pending;
  }
  return SubmissionStatus::Completed;
}

}  // namespace grassland::rhi::backend::vulkan
