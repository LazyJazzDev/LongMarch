#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"

#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"

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

bool VulkanCommandContext::Initialize() {
  const auto device = device_->NativeDevice();
  VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool.queueFamilyIndex = device_->QueueFamily();
  pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  if (!Check(vkCreateCommandPool(device, &pool, nullptr, &pool_), "Create command pool"))
    return false;
  VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (!Check(vkCreateFence(device, &fence, nullptr, &fence_), "Create submission fence"))
    return false;
  VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocate.commandPool = pool_;
  allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate.commandBufferCount = 1;
  if (!Check(vkAllocateCommandBuffers(device, &allocate, &command_), "Allocate command buffer"))
    return false;
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  return Check(vkBeginCommandBuffer(command_, &begin), "Begin command buffer");
}

bool VulkanCommandContext::RecordClear(Image *image, const std::array<float, 4> &color, bool first_use) {
  const auto frame = dynamic_cast<VulkanSwapChainImage *>(image);
  if (!frame || !command_)
    return false;
  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.oldLayout = first_use ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.srcAccessMask = first_use ? 0 : VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = frame->NativeImage();
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  const auto source_stage = first_use ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT;
  vkCmdPipelineBarrier(command_, source_stage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  VkClearColorValue native_color{};
  std::copy(color.begin(), color.end(), native_color.float32);
  vkCmdClearColorImage(command_, barrier.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &native_color, 1,
                       &barrier.subresourceRange);
  return true;
}

bool VulkanCommandContext::RecordPresent(Image *image) {
  const auto frame = dynamic_cast<VulkanSwapChainImage *>(image);
  if (!frame || !command_)
    return false;
  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = frame->NativeImage();
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);
  return true;
}

bool VulkanCommandContext::EndRecording() {
  return Check(vkEndCommandBuffer(command_), "End command buffer");
}

CommandContext::SubmitResult VulkanCommandContext::SubmitNative() {
  if (Frame() && !glfw::IsWindowThread()) {
    ReportError(-1, "Present must run on the window thread.");
    return SUBMIT_RESULT_REJECTED;
  }
  const auto frame = static_cast<VulkanSwapChainImage *>(Frame().Get());
  VkSemaphore acquired{}, rendered{};
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &command_;
  if (frame) {
    acquired = frame->AcquireSemaphore();
    rendered = frame->RenderSemaphore();
    submit.waitSemaphoreCount = submit.signalSemaphoreCount = 1;
    submit.pWaitSemaphores = &acquired;
    submit.pWaitDstStageMask = &stage;
    submit.pSignalSemaphores = &rendered;
  }
  std::lock_guard<std::mutex> lock(device_->QueueMutex());
  if (!Check(vkQueueSubmit(device_->NativeQueue(), 1, &submit, fence_), "vkQueueSubmit"))
    return SUBMIT_RESULT_REJECTED;
  MarkSubmitted();
  failed_ = true;
  failed_ = frame && !Check(frame->Present(), "vkQueuePresentKHR");
  return failed_ ? SUBMIT_RESULT_FAILED_IN_FLIGHT : SUBMIT_RESULT_SUBMITTED;
}

SubmissionStatus VulkanCommandContext::WaitForProgress() {
  const auto result = vkWaitForFences(device_->NativeDevice(), 1, &fence_, VK_TRUE, 50000000);
  if (result == VK_TIMEOUT)
    return SUBMISSION_STATUS_PENDING;
  if (result == VK_ERROR_DEVICE_LOST)
    return SUBMISSION_STATUS_FAILED;
  if (result != VK_SUCCESS)
    failed_ = true;
  if (failed_) {
    std::lock_guard<std::mutex> lock(device_->QueueMutex());
    const auto idle = vkDeviceWaitIdle(device_->NativeDevice());
    return idle == VK_SUCCESS || idle == VK_ERROR_DEVICE_LOST ? SUBMISSION_STATUS_FAILED : SUBMISSION_STATUS_PENDING;
  }
  return SUBMISSION_STATUS_COMPLETED;
}

}  // namespace grassland::rhi::backend::vulkan
