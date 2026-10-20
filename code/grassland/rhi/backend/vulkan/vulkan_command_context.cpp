#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"

#include <cstring>

#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/vulkan/vulkan_buffer.h"
#include "grassland/rhi/backend/vulkan/vulkan_program.h"
#include "grassland/rhi/backend/vulkan/vulkan_swap_chain_image.h"
#include "grassland/rhi/backend/vulkan/vulkan_window.h"

namespace grassland::rhi::backend::vulkan {

VulkanCommandContext::VulkanCommandContext(VulkanDevice *device) : CommandContext(device), device_(device) {
}

VulkanCommandContext::~VulkanCommandContext() {
  if (reusable_) {
    device_->RecycleCommandResources(pool_, command_);
    return;
  }
  if (pool_)
    vkDestroyCommandPool(device_->NativeDevice(), pool_, nullptr);
}

VulkanCommandContext *VulkanCommandContext::Create(VulkanDevice *device) {
  try {
    auto object = std::unique_ptr<VulkanCommandContext>(new VulkanCommandContext(device));
    object->reusable_ = object->device_->AcquireCommandResources(object->pool_, object->command_);
    if (!object->reusable_)
      return nullptr;
    return object.release();
  } catch (const std::bad_alloc &) {
    ReportError(-1, "VulkanCommandContext creation ran out of memory.");
    return nullptr;
  }
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
      barrier.subresourceRange = {ImageAspect(image->GetDesc().format), 0, 1, 0, 1};
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
  if (render_target_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before transfer, clear, present or submission.");
    return false;
  }
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
  if (render_target_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before transfer, clear, present or submission.");
    return false;
  }
  const auto target = dynamic_cast<VulkanWindow *>(window);
  if (!target || target->DeviceIdentity() != device_ || !glfw::IsWindowThread()) {
    recording_failed_ = true;
    ReportError(-1, "CmdPresent requires a window from this device on the window thread.");
    return false;
  }
  for (const auto &present : presents_) {
    if (present == target) {
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
    presents_.push_back(target);
  } catch (const std::bad_alloc &) {
    recording_failed_ = true;
    return false;
  }
  return true;
}

bool VulkanCommandContext::EndRecording() {
#if !defined(NDEBUG)
  if (render_target_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before transfer, clear, present or submission.");
    return false;
  }
#endif
  for (auto &entry : buffer_states_) {
    if (!TransitionBuffer(entry.first, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                          VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT))
      return false;
  }
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
  std::vector<VulkanWindow *> acquired_windows;
  for (const auto &entry : image_states_) {
    // Swapchain acquisition synchronization is distinct from ordinary image barriers.
    if (auto frame = dynamic_cast<VulkanSwapChainImage *>(entry.first)) {
      auto window = frame->Owner();
      if (auto semaphore = window->AcquireSemaphore()) {
        acquired.push_back(semaphore);
        stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        acquired_windows.push_back(window);
      }
    }
  }
  for (const auto &present : presents_)
    rendered.push_back(present->RenderSemaphore());
  rendered.push_back(device_->Timeline());
  std::vector<uint64_t> wait_values(acquired.size(), 0);
  std::vector<uint64_t> signal_values(rendered.size(), 0);
  VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  timeline.waitSemaphoreValueCount = static_cast<uint32_t>(wait_values.size());
  timeline.pWaitSemaphoreValues = wait_values.data();
  timeline.signalSemaphoreValueCount = static_cast<uint32_t>(signal_values.size());
  timeline.pSignalSemaphoreValues = signal_values.data();
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.pNext = &timeline;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &command_;
  submit.waitSemaphoreCount = static_cast<uint32_t>(acquired.size());
  submit.pWaitSemaphores = acquired.data();
  submit.pWaitDstStageMask = stages.data();
  submit.signalSemaphoreCount = static_cast<uint32_t>(rendered.size());
  submit.pSignalSemaphores = rendered.data();
  {
    std::lock_guard<std::mutex> lock(device_->QueueMutex());
    value_ = device_->NextSerial();
    signal_values.back() = value_;
    if (!Check(vkQueueSubmit(device_->NativeQueue(), 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit"))
      return SUBMIT_RESULT_REJECTED;
    MarkSubmitted();
    for (auto window : acquired_windows)
      window->MarkAcquireWaitSubmitted(value_);
    submission_failed_ = true;
    bool presented = true;
    for (const auto &present : presents_) {
      const auto result = present->PresentFrame();
      // OUT_OF_DATE drops presentation, not the accepted GPU work. The window
      // exposes invalidation to the caller; no swapchain is replaced here.
      if (result != VK_ERROR_OUT_OF_DATE_KHR && result != VK_SUBOPTIMAL_KHR && !Check(result, "vkQueuePresentKHR"))
        presented = false;
    }
    submission_failed_ = !presented;
  }
  if (submission_failed_)
    return SUBMIT_RESULT_FAILED_IN_FLIGHT;
  for (const auto &present : presents_) {
    if (present->IsSwapChainInvalid()) {
      present->ReleaseNextImage();
      continue;
    }
    if (!present->UpdateNextImage()) {
      ReportError(-1, "Failed to prepare the next swapchain image after submission.");
      submission_failed_ = true;
    }
  }
  return submission_failed_ ? SUBMIT_RESULT_FAILED_IN_FLIGHT : SUBMIT_RESULT_SUBMITTED;
}

SubmissionStatus VulkanCommandContext::WaitForProgress() {
  const auto result = device_->Wait(value_, 50000000);
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

bool VulkanCommandContext::CmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size) {
#if !defined(NDEBUG)
  if (render_target_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before transfer, clear, present or submission.");
    return false;
  }
  auto native = dynamic_cast<VulkanBuffer *>(buffer);
  if (!native || native->Owner() != device_ || sealed_) {
    recording_failed_ = true;
    return false;
  }
#else
  auto native = static_cast<VulkanBuffer *>(buffer);
#endif
  if (!RetainResource(native))
    return false;

  if (!data || !size || offset > native->GetDesc().size || size > native->GetDesc().size - offset) {
    recording_failed_ = true;
    return false;
  }
  auto upload = device_->UploadPool().Acquire(size);
  if (!upload) {
    recording_failed_ = true;
    return false;
  }
  if (!upload->Write(data, size)) {
    recording_failed_ = true;
    return false;
  }
  if (!RetainResource(upload.Get()))
    return false;
  if (!TransitionBuffer(native, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT))
    return false;
  VkBufferCopy copy{0, offset, size};
  vkCmdCopyBuffer(command_, upload->NativeBuffer(), native->NativeBuffer(), 1, &copy);
  return true;
}

bool VulkanCommandContext::CmdCopyBuffer(IBuffer *source,
                                         uint64_t source_offset,
                                         IBuffer *destination,
                                         uint64_t destination_offset,
                                         uint64_t size) {
#if !defined(NDEBUG)
  if (render_target_) {
    recording_failed_ = true;
    ReportError(-1, "End rendering before transfer, clear, present or submission.");
    return false;
  }
  auto src = dynamic_cast<VulkanBuffer *>(source);
  auto dst = dynamic_cast<VulkanBuffer *>(destination);
  if (!src || !dst || src->Owner() != device_ || dst->Owner() != device_ || sealed_) {
    recording_failed_ = true;
    return false;
  }
#else
  auto src = static_cast<VulkanBuffer *>(source);
  auto dst = static_cast<VulkanBuffer *>(destination);
#endif
  if (src == dst || !size || source_offset > src->GetDesc().size || size > src->GetDesc().size - source_offset ||
      destination_offset > dst->GetDesc().size || size > dst->GetDesc().size - destination_offset) {
    recording_failed_ = true;
    return false;
  }
  if (!RetainResource(src))
    return false;
  if (!RetainResource(dst))
    return false;
  if (!TransitionBuffer(src, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT))
    return false;
  if (!TransitionBuffer(dst, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT))
    return false;
  VkBufferCopy copy{source_offset, destination_offset, size};
  vkCmdCopyBuffer(command_, src->NativeBuffer(), dst->NativeBuffer(), 1, &copy);
  return true;
}

bool VulkanCommandContext::CmdBindProgram(IProgram *program) {
#if !defined(NDEBUG)
  auto native = dynamic_cast<VulkanProgram *>(program);
  if (!native || native->Owner() != device_ || sealed_) {
    recording_failed_ = true;
    return false;
  }
#else
  auto native = static_cast<VulkanProgram *>(program);
#endif
  if (!RetainResource(native))
    return false;
  if (!native->IsFinalized()) {
    recording_failed_ = true;
    return false;
  }
  try {
    vertex_buffers_.assign(native->GetDesc().vertex_bindings.size(), nullptr);
    groups_.assign(native->GetDesc().pipeline_layout.groups.size(), nullptr);
  } catch (const std::bad_alloc &) {
    recording_failed_ = true;
    return false;
  }
  program_ = native;
  vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_GRAPHICS, native->NativePipeline());
  return true;
}

bool VulkanCommandContext::CmdBeginRendering(const RenderingInfo &info) {
  if (sealed_ || render_target_ || !PrepareRendering(info)) {
    recording_failed_ = true;
    return false;
  }
  try {
    for (uint32_t i = 0; i < groups_.size(); ++i)
      if (groups_[i] && !BindGroupVersion(i, groups_[i]))
        return false;
    std::vector<VulkanImage *> images;
    std::vector<ImageFormat> formats;
    for (const auto &color : rendering_.color_attachments) {
      auto image = static_cast<VulkanImage *>(color.color_attachment);
      if (!TransitionImage(image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT))
        return false;
      images.push_back(image);
      formats.push_back(image->GetDesc().format);
    }
    auto depth = static_cast<VulkanImage *>(rendering_.depth_stencil.image);
    if (depth) {
      if (!TransitionImage(depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                           VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT))
        return false;
      images.push_back(depth);
    }
    const auto depth_format = depth ? depth->GetDesc().format : IMAGE_FORMAT_UNKNOWN;
    auto pass = device_->BeginRenderingCommand()
                    ? VK_NULL_HANDLE
                    : device_->RenderTargets().RenderPass(formats, depth_format, &rendering_);
    if (!device_->BeginRenderingCommand() && !pass) {
      recording_failed_ = true;
      return false;
    }
    auto lease = device_->RenderTargets().Acquire(images, pass);
    if (!lease || !RetainResource(lease.Get())) {
      recording_failed_ = true;
      return false;
    }
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
                            VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 1,
                         &barrier, 0, nullptr, 0, nullptr);
    constexpr VkAttachmentLoadOp loads[] = {VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_CLEAR,
                                            VK_ATTACHMENT_LOAD_OP_DONT_CARE};
    constexpr VkAttachmentStoreOp stores[] = {VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_STORE_OP_DONT_CARE};
    std::array<VkClearValue, 9> clear{};
    std::array<VkRenderingAttachmentInfo, 8> colors{};
    for (size_t i = 0; i < formats.size(); ++i) {
      const auto &source = rendering_.color_attachments[i];
      std::copy(source.clear_color.begin(), source.clear_color.end(), clear[i].color.float32);
      colors[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
      colors[i].imageView = lease->Views()[i];
      colors[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      colors[i].loadOp = loads[source.load_op];
      colors[i].storeOp = stores[source.store_op];
      colors[i].clearValue = clear[i];
    }
    const auto &ds = rendering_.depth_stencil;
    VkRenderingAttachmentInfo depth_info{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO},
        stencil_info{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    if (depth) {
      clear[formats.size()].depthStencil = {ds.clear_depth, ds.clear_stencil};
      depth_info.imageView = lease->Views().back();
      depth_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      depth_info.loadOp = loads[ds.depth_load_op];
      depth_info.storeOp = stores[ds.depth_store_op];
      depth_info.clearValue = clear[formats.size()];
      stencil_info = depth_info;
      stencil_info.loadOp = loads[ds.stencil_load_op];
      stencil_info.storeOp = stores[ds.stencil_store_op];
    }
    auto desc = images.front()->GetDesc();
    if (auto begin_rendering = device_->BeginRenderingCommand()) {
      VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
      rendering.renderArea.extent = {desc.width, desc.height};
      rendering.layerCount = 1;
      rendering.colorAttachmentCount = uint32_t(formats.size());
      rendering.pColorAttachments = colors.data();
      rendering.pDepthAttachment = depth ? &depth_info : nullptr;
      rendering.pStencilAttachment = HasStencil(depth_format) ? &stencil_info : nullptr;
      begin_rendering(command_, &rendering);
    } else {
      VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      begin.renderPass = pass;
      begin.framebuffer = lease->Framebuffer();
      begin.renderArea.extent = {desc.width, desc.height};
      begin.clearValueCount = uint32_t(images.size());
      begin.pClearValues = clear.data();
      vkCmdBeginRenderPass(command_, &begin, VK_SUBPASS_CONTENTS_INLINE);
    }
    render_target_ = images.front();
    return true;
  } catch (const std::bad_alloc &) {
    recording_failed_ = true;
    return false;
  }
}

bool VulkanCommandContext::CmdBindVertexBuffer(uint32_t binding, IBuffer *buffer, uint64_t offset) {
#if !defined(NDEBUG)
  auto native = dynamic_cast<VulkanBuffer *>(buffer);
  if (!native || native->Owner() != device_ || sealed_) {
    recording_failed_ = true;
    return false;
  }
#else
  auto native = static_cast<VulkanBuffer *>(buffer);
#endif
  if (!RetainResource(native))
    return false;
  if (!program_ || binding >= program_->GetDesc().vertex_bindings.size() || offset >= native->GetDesc().size) {
    recording_failed_ = true;
    return false;
  }
  if (!TransitionBuffer(native, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT))
    return false;
  auto handle = native->NativeBuffer();
  VkDeviceSize native_offset = offset;
  vkCmdBindVertexBuffers(command_, binding, 1, &handle, &native_offset);
  vertex_buffers_[binding] = native;
  return true;
}

bool VulkanCommandContext::CmdBindIndexBuffer(IBuffer *buffer, IndexFormat format, uint64_t offset) {
#if !defined(NDEBUG)
  auto native = dynamic_cast<VulkanBuffer *>(buffer);
  if (!native || native->Owner() != device_ || sealed_) {
    recording_failed_ = true;
    return false;
  }
#else
  auto native = static_cast<VulkanBuffer *>(buffer);
#endif
  if (!RetainResource(native))
    return false;
  if (format != INDEX_FORMAT_UINT16 && format != INDEX_FORMAT_UINT32) {
    recording_failed_ = true;
    return false;
  }
  auto alignment = format == INDEX_FORMAT_UINT16 ? 2u : 4u;
  if (offset >= native->GetDesc().size || offset % alignment) {
    recording_failed_ = true;
    return false;
  }
  if (!TransitionBuffer(native, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_INDEX_READ_BIT))
    return false;
  vkCmdBindIndexBuffer(command_, native->NativeBuffer(), offset,
                       format == INDEX_FORMAT_UINT16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
  index_bound_ = true;
  return true;
}

bool VulkanCommandContext::CmdSetViewport(const Viewport &viewport) {
#if !defined(NDEBUG)
  if (sealed_ || !viewport.width || !viewport.height) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (recording_failed_)
    return false;
  VkViewport native{viewport.x,       viewport.y + viewport.height, viewport.width,
                    -viewport.height, viewport.min_depth,           viewport.max_depth};
  vkCmdSetViewport(command_, 0, 1, &native);
  viewport_set_ = true;
  return true;
}

bool VulkanCommandContext::CmdSetScissor(const Scissor &scissor) {
#if !defined(NDEBUG)
  if (sealed_ || !scissor.width || !scissor.height) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (recording_failed_)
    return false;
  VkRect2D native{{scissor.x, scissor.y}, {scissor.width, scissor.height}};
  vkCmdSetScissor(command_, 0, 1, &native);
  scissor_set_ = true;
  return true;
}

bool VulkanCommandContext::CmdDraw(uint32_t vertex_count,
                                   uint32_t instance_count,
                                   uint32_t first_vertex,
                                   uint32_t first_instance) {
#if !defined(NDEBUG)
  if (sealed_ || !program_ || !render_target_ || !viewport_set_ || !scissor_set_) {
    recording_failed_ = true;
    return false;
  }
  if (std::find(vertex_buffers_.begin(), vertex_buffers_.end(), nullptr) != vertex_buffers_.end() ||
      !RenderingMatches(program_)) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (recording_failed_)
    return false;
  if (!ValidateGroups())
    return false;
  vkCmdDraw(command_, vertex_count, instance_count, first_vertex, first_instance);
  return true;
}

bool VulkanCommandContext::CmdDrawIndexed(uint32_t index_count,
                                          uint32_t instance_count,
                                          uint32_t first_index,
                                          int32_t vertex_offset,
                                          uint32_t first_instance) {
#if !defined(NDEBUG)
  if (sealed_ || !program_ || !render_target_ || !viewport_set_ || !scissor_set_ || !index_bound_) {
    recording_failed_ = true;
    return false;
  }
  if (std::find(vertex_buffers_.begin(), vertex_buffers_.end(), nullptr) != vertex_buffers_.end() ||
      !RenderingMatches(program_)) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (recording_failed_)
    return false;
  if (!ValidateGroups())
    return false;
  vkCmdDrawIndexed(command_, index_count, instance_count, first_index, vertex_offset, first_instance);
  return true;
}

bool VulkanCommandContext::CmdEndRendering() {
#if !defined(NDEBUG)
  if (sealed_ || !render_target_) {
    recording_failed_ = true;
    return false;
  }
#endif
  if (recording_failed_)
    return false;
  if (auto end_rendering = device_->EndRenderingCommand())
    end_rendering(command_);
  else
    vkCmdEndRenderPass(command_);
  // Make reads in this scope complete before later buffer writes, including buffers
  // first bound inside the scope. Buffer bindings do not change image layouts.
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT |
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier,
                       0, nullptr, 0, nullptr);
  render_target_ = nullptr;
  for (auto color : rendering_.color_attachments) {
    auto image = static_cast<VulkanImage *>(color.color_attachment);
    if (!TransitionImage(image, image->DefaultLayout(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT))
      return false;
  }
  return true;
}

bool VulkanCommandContext::TransitionBuffer(VulkanBuffer *buffer, VkPipelineStageFlags stages, VkAccessFlags access) {
  if (render_target_)
    return true;  // Graphics buffer accesses are synchronized by the explicit scope boundaries.
  auto known = buffer_states_.find(buffer);
  constexpr VkAccessFlags reads =
      VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  if (known != buffer_states_.end() && (known->second.access & ~reads) == 0 && (access & ~reads) == 0) {
    if ((access & ~known->second.access) == 0 && (stages & ~known->second.stages) == 0)
      return true;
    // A new read usage needs visibility of earlier writes too (e.g. vertex -> index).
    access |= known->second.access;
    stages |= known->second.stages;
  }
  try {
    auto &state = buffer_states_
                      .try_emplace(buffer, BufferState{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                                       VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT})
                      .first->second;
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = access;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer->NativeBuffer();
    barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(command_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, stages, 0, 0, nullptr, 1, &barrier, 0, nullptr);
    state = {stages, access};
    return true;
  } catch (const std::bad_alloc &) {
    recording_failed_ = true;
    return false;
  }
}

bool VulkanCommandContext::CmdUploadImage(IImage *image, const void *data, uint64_t row_pitch) {
  auto native = dynamic_cast<VulkanImage *>(image);
  if (sealed_ || render_target_ || !native || !native->IsValid() || native->DeviceIdentity() != device_ || !data ||
      IsDepthFormat(native->GetDesc().format) || row_pitch < uint64_t(native->GetDesc().width) * 4 ||
      (native->GetDesc().height - 1) > (SIZE_MAX - uint64_t(native->GetDesc().width) * 4) / row_pitch) {
    recording_failed_ = true;
    return false;
  }
  try {
    const auto desc = native->GetDesc();
    const uint64_t packed = uint64_t(desc.width) * 4;
    std::vector<uint8_t> pixels(packed * desc.height);
    for (uint32_t y = 0; y < desc.height; ++y)
      std::memcpy(pixels.data() + y * packed, static_cast<const uint8_t *>(data) + y * row_pitch, packed);
    auto upload = device_->UploadPool().Acquire(pixels.size());
    if (!upload || !upload->Write(pixels.data(), pixels.size()) || !RetainResource(upload.Get()) ||
        !RetainResource(native)) {
      recording_failed_ = true;
      return false;
    }
    if (!TransitionImage(native, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_ACCESS_TRANSFER_WRITE_BIT))
      return false;
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {desc.width, desc.height, 1};
    vkCmdCopyBufferToImage(command_, upload->NativeBuffer(), native->NativeImage(),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    return TransitionImage(native, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                           VK_ACCESS_SHADER_READ_BIT);
  } catch (const std::bad_alloc &) {
    recording_failed_ = true;
    return false;
  }
}

bool VulkanCommandContext::CmdBindGroup(uint32_t index, IBindingGroup *group) {
  auto native = dynamic_cast<VulkanBindingGroup *>(group);
  if (sealed_ || !program_ || !native || native->Owner() != device_ || index >= groups_.size() ||
      !SameBindingLayout(native->GetLayout(), program_->GetDesc().pipeline_layout.groups[index]) ||
      !native->CurrentVersion()->IsComplete()) {
    recording_failed_ = true;
    return false;
  }
  auto version = static_cast<VulkanBindingGroup::Version *>(native->CurrentVersion());
  if (!RetainResource(version))
    return false;
  return BindGroupVersion(index, version);
}

bool VulkanCommandContext::BindGroupVersion(uint32_t index, VulkanBindingGroup::Version *native) {
  try {
    size_t entry = 0;
    for (auto b : native->GetLayout().bindings)
      for (uint32_t i = 0; i < b.count; ++i) {
        auto e = native->GetResources()[entry++];
        const VkAccessFlags access =
            VK_ACCESS_SHADER_READ_BIT |
            ((b.type == BINDING_RW_STORAGE_BUFFER || b.type == BINDING_RW_TEXTURE2D) ? VK_ACCESS_SHADER_WRITE_BIT : 0);
        if (e.buffer && !TransitionBuffer(static_cast<VulkanBuffer *>(e.buffer), VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                                          b.type == BINDING_UNIFORM_BUFFER ? VK_ACCESS_UNIFORM_READ_BIT : access))
          return false;
        if (e.image) {
          auto image = static_cast<VulkanImage *>(e.image);
          if (render_target_) {
            auto known = image_states_.find(image);
            if ((known != image_states_.end() && known->second.layout != VK_IMAGE_LAYOUT_GENERAL) ||
                image->DefaultLayout() != VK_IMAGE_LAYOUT_GENERAL) {
              recording_failed_ = true;
              return false;
            }
            auto &state = image_states_.try_emplace(image, ImageState{VK_IMAGE_LAYOUT_GENERAL, 0, 0}).first->second;
            state.stages |= VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
            state.access |= access;
          } else if (!TransitionImage(image, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, access))
            return false;
        }
      }
    groups_[index] = native;
    auto set = native->Set();
    vkCmdBindDescriptorSets(command_, VK_PIPELINE_BIND_POINT_GRAPHICS, program_->NativeLayout(), index, 1, &set, 0,
                            nullptr);
    return true;
  } catch (const std::bad_alloc &) {
    recording_failed_ = true;
    return false;
  }
}

bool VulkanCommandContext::ValidateGroups() {
  for (size_t i = 0; i < groups_.size(); ++i) {
    if (!groups_[i] && !program_->GetDesc().pipeline_layout.groups[i].bindings.empty()) {
      recording_failed_ = true;
      return false;
    }
    if (groups_[i])
      for (auto e : groups_[i]->GetResources())
        if (e.image) {
          for (auto c : rendering_.color_attachments)
            if (c.color_attachment == e.image) {
              recording_failed_ = true;
              return false;
            }
        }
  }
  return !recording_failed_;
}

}  // namespace grassland::rhi::backend::vulkan
