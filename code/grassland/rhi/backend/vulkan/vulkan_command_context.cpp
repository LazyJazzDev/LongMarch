#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"

#include <cstdlib>

#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_window.h"

namespace grassland::rhi::backend::vulkan {

VulkanCommandContext::VulkanCommandContext(VulkanDevice *device) : CommandContext(device), device_(device) {
}

VulkanCommandContext::~VulkanCommandContext() {
  if (reusable_) {
    device_->RecycleCommandResources(pool_, command_, fence_);
    return;
  }
  if (fence_)
    vkDestroyFence(device_->NativeDevice(), fence_, nullptr);
  if (pool_)
    vkDestroyCommandPool(device_->NativeDevice(), pool_, nullptr);
}

bool VulkanCommandContext::Initialize() {
  reusable_ = device_->AcquireCommandResources(pool_, command_, fence_);
  return reusable_;
}

bool VulkanCommandContext::TransitionImage(VulkanImage *image,
                                           VkImageLayout layout,
                                           VkPipelineStageFlags stages,
                                           VkAccessFlags access) {
  try {
    const auto default_access = image->DefaultLayout() == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                    ? VkAccessFlags{0}
                                    : VkAccessFlags{VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    auto &current =
        image_states_
            .try_emplace(image, ImageState{image->DefaultLayout(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, default_access})
            .first->second;
    constexpr VkAccessFlags writes = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                                     VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    // Equal layouts still need a memory dependency for write-after-write/read uses.
    if (current.layout != layout || (current.access & writes) || (current.access && (access & writes))) {
      VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      barrier.oldLayout = current.layout;
      barrier.newLayout = layout;
      barrier.srcAccessMask = current.access;
      barrier.dstAccessMask = access;
      barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = image->NativeImage();
      barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      vkCmdPipelineBarrier(command_, current.stages, stages, 0, 0, nullptr, 0, nullptr, 1, &barrier);
      current = {layout, stages, access};
    } else {
      current.stages |= stages;
      current.access |= access;
    }
    return true;
  } catch (const std::bad_alloc &) {
    recording_failed_ = true;
    ReportError(-1, "Image state tracking ran out of memory.");
    return false;
  }
}

bool VulkanCommandContext::CmdClearColor(rhi::IImage *image, const std::array<float, 4> &color) {
#if !defined(NDEBUG)
  const auto native = dynamic_cast<VulkanImage *>(image);
  if (!native || !native->IsValid() || native->DeviceIdentity() != device_) {
    recording_failed_ = true;
    ReportError(-1, "CmdClearColor requires a valid image from this device.");
    return false;
  }
#else
  const auto native = static_cast<VulkanImage *>(image);
#endif
  if (!RetainResource(native))
    return false;
  if (!TransitionImage(native, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_TRANSFER_WRITE_BIT))
    return false;
  VkClearColorValue native_color{};
  std::copy(color.begin(), color.end(), native_color.float32);
  VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdClearColorImage(command_, native->NativeImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &native_color, 1, &range);
  return true;
}

bool VulkanCommandContext::CmdPresent(rhi::IWindow *window) {
#if !defined(NDEBUG)
  const auto target = dynamic_cast<VulkanWindow *>(window);
  if (!target || target->DeviceIdentity() != device_ || !glfw::IsWindowThread()) {
    recording_failed_ = true;
    ReportError(-1, "CmdPresent requires a window from this device on the window thread.");
    return false;
  }
  for (const auto &present : presents_) {
    if (present.first == target) {
      recording_failed_ = true;
      ReportError(-1, "A context can present each window only once.");
      return false;
    }
  }
#else
  const auto target = static_cast<VulkanWindow *>(window);
#endif
  auto frame = ObjectPtr<VulkanSwapChainImage>::Adopt(static_cast<VulkanSwapChainImage *>(target->GetNextImage()));
  if (!frame)
    return false;
  if (!RetainResource(frame.Get()))
    return false;
  if (!TransitionImage(frame.Get(), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0))
    return false;
  try {
    presents_.emplace_back(target, frame.Get());
  } catch (const std::bad_alloc &) {
    recording_failed_ = true;
    return false;
  }
  return true;
}

bool VulkanCommandContext::EndRecording() {
  for (auto &entry : image_states_) {
    const auto layout = entry.first->DefaultLayout();
    const auto access = layout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                            ? VkAccessFlags{0}
                            : VkAccessFlags{VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    if (!TransitionImage(entry.first, layout, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, access))
      return false;
  }
  return Check(vkEndCommandBuffer(command_), "End command buffer");
}

CommandContext::SubmitResult VulkanCommandContext::SubmitNative() {
#if !defined(NDEBUG)
  if (!presents_.empty() && !glfw::IsWindowThread()) {
    ReportError(-1, "Present must run on the window thread.");
    return SUBMIT_RESULT_REJECTED;
  }
  for (const auto &entry : image_states_) {
    if (!entry.first->IsValid())
      return SUBMIT_RESULT_REJECTED;
  }
#endif
  std::vector<VkSemaphore> acquired, rendered;
  std::vector<VkPipelineStageFlags> stages;
  std::vector<VulkanSwapChainImage *> acquired_frames;
  for (const auto &entry : image_states_) {
    // Swapchain acquisition synchronization is distinct from ordinary image barriers.
    if (auto frame = dynamic_cast<VulkanSwapChainImage *>(entry.first)) {
      if (auto semaphore = frame->AcquireSemaphore()) {
        acquired.push_back(semaphore);
        stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        acquired_frames.push_back(frame);
      }
    }
  }
  for (const auto &present : presents_)
    rendered.push_back(present.second->RenderSemaphore());
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &command_;
  submit.waitSemaphoreCount = static_cast<uint32_t>(acquired.size());
  submit.pWaitSemaphores = acquired.data();
  submit.pWaitDstStageMask = stages.data();
  submit.signalSemaphoreCount = static_cast<uint32_t>(rendered.size());
  submit.pSignalSemaphores = rendered.data();
  {
    std::lock_guard<std::mutex> lock(device_->QueueMutex());
    if (!Check(vkQueueSubmit(device_->NativeQueue(), 1, &submit, fence_), "vkQueueSubmit"))
      return SUBMIT_RESULT_REJECTED;
    MarkSubmitted();
    for (auto frame : acquired_frames)
      frame->ConsumeAcquireSemaphore();
    submission_failed_ = true;
    bool presented = true;
    for (const auto &present : presents_) {
      if (!Check(present.second->Present(), "vkQueuePresentKHR"))
        presented = false;
    }
    submission_failed_ = !presented;
  }
  if (submission_failed_)
    return SUBMIT_RESULT_FAILED_IN_FLIGHT;
  for (const auto &present : presents_) {
    if (!present.first->UpdateNextImage()) {
      ReportError(-1, "Failed to prepare the next swapchain image after submission.");
      std::abort();
    }
  }
  return SUBMIT_RESULT_SUBMITTED;
}

SubmissionStatus VulkanCommandContext::WaitForProgress() {
  const auto result = vkWaitForFences(device_->NativeDevice(), 1, &fence_, VK_TRUE, 50000000);
  if (result == VK_TIMEOUT)
    return SUBMISSION_STATUS_PENDING;
  if (result == VK_ERROR_DEVICE_LOST)
    return SUBMISSION_STATUS_FAILED;
  if (result != VK_SUCCESS)
    submission_failed_ = true;
  if (submission_failed_) {
    std::lock_guard<std::mutex> lock(device_->QueueMutex());
    const auto idle = vkDeviceWaitIdle(device_->NativeDevice());
    return idle == VK_SUCCESS || idle == VK_ERROR_DEVICE_LOST ? SUBMISSION_STATUS_FAILED : SUBMISSION_STATUS_PENDING;
  }
  return SUBMISSION_STATUS_COMPLETED;
}

}  // namespace grassland::rhi::backend::vulkan
