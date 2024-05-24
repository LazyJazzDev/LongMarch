#include "grassland/vulkan/core/core.h"

#include "grassland/vulkan/image.h"

grassland::vulkan::Core::Core(const grassland::vulkan::CoreSettings &settings)
    : settings_(settings) {
  InstanceCreateHint hint{settings_.window != nullptr};
  CreateInstance(hint, &instance_);
  if (settings_.window) {
    instance_->CreateSurfaceFromGLFWWindow(settings_.window, &surface_);
  }
  instance_->CreateDevice(surface_.get(), settings.enable_ray_tracing,
                          settings.device_index, &device_);
  device_->GetQueue(device_->PhysicalDevice().GraphicsFamilyIndex(), 0,
                    &graphics_queue_);
  device_->GetQueue(device_->PhysicalDevice().TransferFamilyIndex(), 0,
                    &transfer_queue_);

  if (settings_.window) {
    device_->GetQueue(
        device_->PhysicalDevice().PresentFamilyIndex(surface_.get()), 0,
        &present_queue_);

    device_->CreateSwapchain(surface_.get(), &swap_chain_);
  }

  device_->CreateCommandPool(device_->PhysicalDevice().GraphicsFamilyIndex(),
                             VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                             &graphics_command_pool_);
  device_->CreateCommandPool(device_->PhysicalDevice().TransferFamilyIndex(),
                             VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                             &transfer_command_pool_);
  command_buffers_.resize(settings_.max_frames_in_flight);
  in_flight_fences_.resize(settings_.max_frames_in_flight);

  for (int i = 0; i < settings_.max_frames_in_flight; i++) {
    graphics_command_pool_->AllocateCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, &command_buffers_[i]);
    device_->CreateFence(true, &in_flight_fences_[i]);
  }

  if (settings_.window) {
    image_available_semaphores_.resize(settings_.max_frames_in_flight);
    render_finish_semaphores_.resize(settings_.max_frames_in_flight);

    for (int i = 0; i < settings_.max_frames_in_flight; i++) {
      device_->CreateSemaphore(&image_available_semaphores_[i]);
      device_->CreateSemaphore(&render_finish_semaphores_[i]);
    }
  }
}

grassland::vulkan::Core::~Core() {
  device_->WaitIdle();

  image_available_semaphores_.clear();
  render_finish_semaphores_.clear();
  in_flight_fences_.clear();

  swap_chain_.reset();
  command_buffers_.clear();

  graphics_queue_.reset();
  transfer_queue_.reset();
  present_queue_.reset();

  graphics_command_pool_.reset();
  transfer_command_pool_.reset();

  device_.reset();
  surface_.reset();
  instance_.reset();
}

void grassland::vulkan::Core::RebuildSwapChain() {
  device_->WaitIdle();
  swap_chain_.reset();
  device_->CreateSwapchain(surface_.get(), &swap_chain_);
}

void grassland::vulkan::Core::SingleTimeCommands(
    const std::function<void(VkCommandBuffer)> &function) {
  graphics_command_pool_->SingleTimeCommands(graphics_queue_.get(), function);
}

void grassland::vulkan::Core::BeginFrame() {
  // Wait for current frame fence
  VkFence fence = in_flight_fences_[current_frame_]->Handle();
  vkWaitForFences(device_->Handle(), 1, &fence, VK_TRUE, UINT64_MAX);
  vkResetFences(device_->Handle(), 1, &fence);

  if (settings_.window) {
    // Acquire next image
    VkSemaphore image_available_semaphore =
        image_available_semaphores_[current_frame_]->Handle();
    VkResult result = swap_chain_->AcquireNextImage(
        std::numeric_limits<uint64_t>::max(), image_available_semaphore,
        VK_NULL_HANDLE, &image_index_);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
      // Recreate swap chain
      RebuildSwapChain();
      LogError("Swap chain out of date");
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
      LogError("Failed to acquire swap chain image");
    }
  }

  // Reset command buffer
  VkCommandBuffer command_buffer = command_buffers_[current_frame_]->Handle();

  vkResetCommandBuffer(command_buffer, 0);

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
    LogError("Failed to begin recording command buffer");
  }
}

void grassland::vulkan::Core::EndFrame() {
  VkCommandBuffer command_buffer = command_buffers_[current_frame_]->Handle();
  if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
    LogError("Failed to record command buffer");
  }

  // Submit command buffer
  VkSemaphore image_available_semaphore;
  VkSemaphore render_finish_semaphore;

  if (settings_.window) {
    image_available_semaphore =
        image_available_semaphores_[current_frame_]->Handle();
    render_finish_semaphore =
        render_finish_semaphores_[current_frame_]->Handle();
  }

  VkFence fence = in_flight_fences_[current_frame_]->Handle();
  VkPipelineStageFlags wait_stages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.pWaitDstStageMask = wait_stages;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;

  if (settings_.window) {
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &image_available_semaphore;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &render_finish_semaphore;
  }

  if (vkQueueSubmit(graphics_queue_->Handle(), 1, &submit_info, fence) !=
      VK_SUCCESS) {
    LogError("Failed to submit draw command buffer");
  }
  // Present image
  if (settings_.window) {
    VkSwapchainKHR swap_chain = swap_chain_->Handle();
    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &render_finish_semaphore;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swap_chain;
    present_info.pImageIndices = &image_index_;
    present_info.pResults = nullptr;
    VkResult result =
        vkQueuePresentKHR(present_queue_->Handle(), &present_info);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
      // Recreate swap chain
      RebuildSwapChain();
      LogInfo("Swap chain out of date");
    } else if (result != VK_SUCCESS) {
      LogError("Failed to present swap chain image");
    }
  }

  current_frame_ = (current_frame_ + 1) % settings_.max_frames_in_flight;
}

void grassland::vulkan::Core::OutputFrame(grassland::vulkan::Image *image) {
  VkImage swapchain_image = swap_chain_->Image(image_index_);
  TransitImageLayout(
      command_buffers_[current_frame_]->Handle(), swapchain_image,
      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
      VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
  TransitImageLayout(command_buffers_[current_frame_]->Handle(),
                     image->Handle(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT,
                     VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

  VkImageBlit blit{};
  blit.srcOffsets[0] = {0, 0, 0};
  blit.srcOffsets[1] = {static_cast<int32_t>(image->Width()),
                        static_cast<int32_t>(image->Height()), 1};
  blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  blit.srcSubresource.mipLevel = 0;
  blit.srcSubresource.baseArrayLayer = 0;
  blit.srcSubresource.layerCount = 1;
  blit.dstOffsets[0] = {0, 0, 0};
  blit.dstOffsets[1] = {static_cast<int32_t>(swap_chain_->Extent().width),
                        static_cast<int32_t>(swap_chain_->Extent().height), 1};
  blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  blit.dstSubresource.mipLevel = 0;
  blit.dstSubresource.baseArrayLayer = 0;
  blit.dstSubresource.layerCount = 1;

  vkCmdBlitImage(command_buffers_[current_frame_]->Handle(), image->Handle(),
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchain_image,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                 VK_FILTER_LINEAR);

  TransitImageLayout(
      command_buffers_[current_frame_]->Handle(), swapchain_image,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_IMAGE_ASPECT_COLOR_BIT);
  TransitImageLayout(
      command_buffers_[current_frame_]->Handle(), image->Handle(),
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
      VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
}
