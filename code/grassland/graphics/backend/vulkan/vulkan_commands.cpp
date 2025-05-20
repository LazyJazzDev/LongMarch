#include "grassland/graphics/backend/vulkan/vulkan_commands.h"

#include "grassland/graphics/backend/vulkan/vulkan_acceleration_structure.h"
#include "grassland/graphics/backend/vulkan/vulkan_buffer.h"
#include "grassland/graphics/backend/vulkan/vulkan_command_context.h"
#include "grassland/graphics/backend/vulkan/vulkan_image.h"
#include "grassland/graphics/backend/vulkan/vulkan_program.h"
#include "grassland/graphics/backend/vulkan/vulkan_sampler.h"
#include "grassland/graphics/backend/vulkan/vulkan_window.h"

namespace grassland::graphics::backend {

VulkanCmdBindProgram::VulkanCmdBindProgram(VulkanProgram *program) : program_(program) {
}

void VulkanCmdBindProgram::CompileCommand(VulkanCommandContext *context, VkCommandBuffer command_buffer) {
  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, program_->Pipeline()->Handle());
}

VulkanCmdBindRayTracingProgram::VulkanCmdBindRayTracingProgram(VulkanRayTracingProgram *program) : program_(program) {
}

void VulkanCmdBindRayTracingProgram::CompileCommand(VulkanCommandContext *context, VkCommandBuffer command_buffer) {
  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, program_->Pipeline()->Handle());
}

VulkanCmdBindVertexBuffers::VulkanCmdBindVertexBuffers(uint32_t first_binding,
                                                       const std::vector<VulkanBuffer *> &buffers,
                                                       const std::vector<uint64_t> &offsets)
    : first_binding_(first_binding), buffers_(buffers) {
  offsets_.resize(buffers_.size());
  for (size_t i = 0; i < buffers_.size(); ++i) {
    if (i < offsets.size()) {
      offsets_[i] = offsets[i];
    } else {
      offsets_[i] = 0;
    }
  }
}

void VulkanCmdBindVertexBuffers::CompileCommand(VulkanCommandContext *context, VkCommandBuffer command_buffer) {
  std::vector<VkBuffer> buffers(buffers_.size());
  for (size_t i = 0; i < buffers_.size(); ++i) {
    buffers[i] = buffers_[i]->Buffer();
  }
  vkCmdBindVertexBuffers(command_buffer, first_binding_, buffers.size(), buffers.data(), offsets_.data());
}

VulkanCmdBindIndexBuffer::VulkanCmdBindIndexBuffer(VulkanBuffer *buffer, uint64_t offset)
    : buffer_(buffer), offset_(offset) {
}

void VulkanCmdBindIndexBuffer::CompileCommand(VulkanCommandContext *context, VkCommandBuffer command_buffer) {
  vkCmdBindIndexBuffer(command_buffer, buffer_->Buffer(), offset_, VK_INDEX_TYPE_UINT32);
}

VulkanCmdBeginRendering::VulkanCmdBeginRendering(const std::vector<VulkanImage *> &color_targets,
                                                 VulkanImage *depth_target)
    : color_targets_(color_targets), depth_target_(depth_target) {
}

void VulkanCmdBeginRendering::CompileCommand(VulkanCommandContext *context, VkCommandBuffer command_buffer) {
  for (auto &image : resource_images_) {
    context->RequireImageState(command_buffer, image->Image()->Handle(), VK_IMAGE_LAYOUT_GENERAL,
                               VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                               VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, image->Image()->Aspect());
  }

  std::vector<VkRenderingAttachmentInfo> color_attachment_infos;
  VkRenderingAttachmentInfo depth_attachment_info{};
  Extent2D extent;
  extent.width = 0x7fffffff;
  extent.height = 0x7fffffff;
  for (int i = 0; i < color_targets_.size(); i++) {
    auto &color_target = color_targets_[i];
    context->RequireImageState(command_buffer, color_target->Image()->Handle(),
                               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, color_target->Image()->Aspect());

    VkRenderingAttachmentInfo attachment_info{};
    attachment_info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    attachment_info.pNext = nullptr;
    attachment_info.imageView = color_target->Image()->ImageView();
    attachment_info.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment_infos.push_back(attachment_info);

    auto target_extent = color_target->Extent();
    extent.width = std::min(extent.width, target_extent.width);
    extent.height = std::min(extent.height, target_extent.height);
  }
  VkRenderingInfo rendering_info{};
  rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
  rendering_info.pNext = nullptr;
  rendering_info.colorAttachmentCount = color_attachment_infos.size();
  rendering_info.pColorAttachments = color_attachment_infos.data();
  rendering_info.renderArea.offset = {0, 0};
  rendering_info.layerCount = 1;
  if (depth_target_) {
    context->RequireImageState(command_buffer, depth_target_->Image()->Handle(),
                               VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                               depth_target_->Image()->Aspect());
    depth_attachment_info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_attachment_info.pNext = nullptr;
    depth_attachment_info.imageView = depth_target_->Image()->ImageView();
    depth_attachment_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depth_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    rendering_info.pDepthAttachment = &depth_attachment_info;
    auto target_extent = depth_target_->Extent();
    extent.width = std::min(extent.width, target_extent.width);
    extent.height = std::min(extent.height, target_extent.height);
  }
  rendering_info.renderArea.extent.width = extent.width;
  rendering_info.renderArea.extent.height = extent.height;
  context->Core()->Instance()->Procedures().vkCmdBeginRenderingKHR(command_buffer, &rendering_info);
}

void VulkanCmdBeginRendering::RecordResourceImages(VulkanImage *resource_image) {
  resource_images_.insert(resource_image);
}

VulkanCmdBindResourceBuffers::VulkanCmdBindResourceBuffers(int slot,
                                                           const std::vector<VulkanBuffer *> &buffers,
                                                           VulkanProgramBase *program_base,
                                                           BindPoint bind_point)
    : slot_(slot), buffers_(buffers), program_base_(program_base), bind_point_(bind_point) {
}

void VulkanCmdBindResourceBuffers::CompileCommand(VulkanCommandContext *context, VkCommandBuffer command_buffer) {
  std::vector<VkDescriptorBufferInfo> buffer_infos(buffers_.size());
  for (size_t i = 0; i < buffers_.size(); ++i) {
    buffer_infos[i].buffer = buffers_[i]->Buffer();
    buffer_infos[i].offset = 0;
    buffer_infos[i].range = buffers_[i]->Size();
  }
  auto descriptor_set = context->AcquireDescriptorSet(program_base_->DescriptorSetLayout(slot_)->Handle());

  auto binding = program_base_->DescriptorSetLayout(slot_)->Bindings()[0];

  VkWriteDescriptorSet write_descriptor_set{};
  write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_descriptor_set.dstSet = descriptor_set->Handle();
  write_descriptor_set.dstBinding = binding.binding;
  write_descriptor_set.dstArrayElement = 0;
  write_descriptor_set.descriptorCount = binding.descriptorCount;
  write_descriptor_set.descriptorType = binding.descriptorType;
  write_descriptor_set.pBufferInfo = buffer_infos.data();
  vkUpdateDescriptorSets(context->Core()->Device()->Handle(), 1, &write_descriptor_set, 0, nullptr);

  VkDescriptorSet descriptor_sets[] = {descriptor_set->Handle()};

  vkCmdBindDescriptorSets(command_buffer, BindPointToVkPipelineBindPoint(bind_point_),
                          program_base_->PipelineLayout()->Handle(), slot_, 1, descriptor_sets, 0, nullptr);
}

VulkanCmdBindResourceImages::VulkanCmdBindResourceImages(int slot,
                                                         const std::vector<VulkanImage *> &images,
                                                         VulkanProgramBase *program_base,
                                                         BindPoint bind_point,
                                                         bool update_layout)
    : slot_(slot),
      images_(images),
      program_base_(program_base),
      bind_point_(bind_point),
      update_layout_(update_layout) {
}

void VulkanCmdBindResourceImages::CompileCommand(VulkanCommandContext *context, VkCommandBuffer command_buffer) {
  std::vector<VkDescriptorImageInfo> image_infos(images_.size());
  for (size_t i = 0; i < images_.size(); ++i) {
    image_infos[i].sampler = VK_NULL_HANDLE;
    image_infos[i].imageView = images_[i]->Image()->ImageView();
    image_infos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    if (update_layout_) {
      context->RequireImageState(command_buffer, images_[i]->Image()->Handle(), VK_IMAGE_LAYOUT_GENERAL,
                                 VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                                 VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, images_[i]->Image()->Aspect());
    }
  }
  auto descriptor_set = context->AcquireDescriptorSet(program_base_->DescriptorSetLayout(slot_)->Handle());
  VkDescriptorSet descriptor_sets[] = {descriptor_set->Handle()};
  auto binding = program_base_->DescriptorSetLayout(slot_)->Bindings()[0];
  VkWriteDescriptorSet write_descriptor_set{};
  write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_descriptor_set.dstSet = descriptor_set->Handle();
  write_descriptor_set.dstBinding = binding.binding;
  write_descriptor_set.dstArrayElement = 0;
  write_descriptor_set.descriptorCount = binding.descriptorCount;
  write_descriptor_set.descriptorType = binding.descriptorType;
  write_descriptor_set.pImageInfo = image_infos.data();
  vkUpdateDescriptorSets(context->Core()->Device()->Handle(), 1, &write_descriptor_set, 0, nullptr);

  vkCmdBindDescriptorSets(command_buffer, BindPointToVkPipelineBindPoint(bind_point_),
                          program_base_->PipelineLayout()->Handle(), slot_, 1, descriptor_sets, 0, nullptr);
}

VulkanCmdBindResourceSamplers::VulkanCmdBindResourceSamplers(int slot,
                                                             const std::vector<VulkanSampler *> &samplers,
                                                             VulkanProgramBase *program_base,
                                                             BindPoint bind_point)
    : slot_(slot), samplers_(samplers), program_base_(program_base), bind_point_(bind_point) {
}

void VulkanCmdBindResourceSamplers::CompileCommand(VulkanCommandContext *context, VkCommandBuffer command_buffer) {
  std::vector<VkDescriptorImageInfo> sampler_infos(samplers_.size());
  for (size_t i = 0; i < samplers_.size(); ++i) {
    sampler_infos[i].sampler = samplers_[i]->Sampler()->Handle();
    sampler_infos[i].imageView = VK_NULL_HANDLE;
    sampler_infos[i].imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  }
  auto descriptor_set = context->AcquireDescriptorSet(program_base_->DescriptorSetLayout(slot_)->Handle());
  VkDescriptorSet descriptor_sets[] = {descriptor_set->Handle()};
  auto binding = program_base_->DescriptorSetLayout(slot_)->Bindings()[0];
  VkWriteDescriptorSet write_descriptor_set{};
  write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_descriptor_set.dstSet = descriptor_set->Handle();
  write_descriptor_set.dstBinding = binding.binding;
  write_descriptor_set.dstArrayElement = 0;
  write_descriptor_set.descriptorCount = binding.descriptorCount;
  write_descriptor_set.descriptorType = binding.descriptorType;
  write_descriptor_set.pImageInfo = sampler_infos.data();
  vkUpdateDescriptorSets(context->Core()->Device()->Handle(), 1, &write_descriptor_set, 0, nullptr);

  vkCmdBindDescriptorSets(command_buffer, BindPointToVkPipelineBindPoint(bind_point_),
                          program_base_->PipelineLayout()->Handle(), slot_, 1, descriptor_sets, 0, nullptr);
}

VulkanCmdBindResourceAccelerationStructure::VulkanCmdBindResourceAccelerationStructure(
    int slot,
    VulkanAccelerationStructure *acceleration_structure,
    VulkanProgramBase *program_base,
    BindPoint bind_point)
    : slot_(slot),
      acceleration_structure_(acceleration_structure),
      program_base_(program_base),
      bind_point_(bind_point) {
}

void VulkanCmdBindResourceAccelerationStructure::CompileCommand(VulkanCommandContext *context,
                                                                VkCommandBuffer command_buffer) {
  auto descriptor_set = context->AcquireDescriptorSet(program_base_->DescriptorSetLayout(slot_)->Handle());
  VkDescriptorSet descriptor_sets[] = {descriptor_set->Handle()};
  auto binding = program_base_->DescriptorSetLayout(slot_)->Bindings()[0];
  VkWriteDescriptorSet write_descriptor_set{};
  write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_descriptor_set.dstSet = descriptor_set->Handle();
  write_descriptor_set.dstBinding = binding.binding;
  write_descriptor_set.dstArrayElement = 0;
  write_descriptor_set.descriptorCount = binding.descriptorCount;
  write_descriptor_set.descriptorType = binding.descriptorType;

  VkAccelerationStructureKHR acceleration_structure = acceleration_structure_->Handle()->Handle();

  VkWriteDescriptorSetAccelerationStructureKHR acceleration_structure_info{};
  acceleration_structure_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
  acceleration_structure_info.pNext = nullptr;
  acceleration_structure_info.accelerationStructureCount = 1;
  acceleration_structure_info.pAccelerationStructures = &acceleration_structure;
  write_descriptor_set.pNext = &acceleration_structure_info;

  vkUpdateDescriptorSets(context->Core()->Device()->Handle(), 1, &write_descriptor_set, 0, nullptr);

  vkCmdBindDescriptorSets(command_buffer, BindPointToVkPipelineBindPoint(bind_point_),
                          program_base_->PipelineLayout()->Handle(), slot_, 1, descriptor_sets, 0, nullptr);
}

void VulkanCmdEndRendering::CompileCommand(VulkanCommandContext *context, VkCommandBuffer command_buffer) {
  context->Core()->Instance()->Procedures().vkCmdEndRenderingKHR(command_buffer);
}

VulkanCmdClearImage::VulkanCmdClearImage(VulkanImage *image, const ClearValue &clear_value)
    : image_(image), clear_value_(clear_value) {
}

void VulkanCmdClearImage::CompileCommand(VulkanCommandContext *context, VkCommandBuffer command_buffer) {
  context->RequireImageState(command_buffer, image_->Image()->Handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, image_->Image()->Aspect());
  if (!vulkan::IsDepthFormat(image_->Image()->Format())) {
    VkClearColorValue clear_value;
    clear_value.float32[0] = clear_value_.color.r;
    clear_value.float32[1] = clear_value_.color.g;
    clear_value.float32[2] = clear_value_.color.b;
    clear_value.float32[3] = clear_value_.color.a;
    VkImageSubresourceRange subresource_range;
    subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource_range.baseMipLevel = 0;
    subresource_range.levelCount = 1;
    subresource_range.baseArrayLayer = 0;
    subresource_range.layerCount = 1;
    vkCmdClearColorImage(command_buffer, image_->Image()->Handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_value,
                         1, &subresource_range);
  } else {
    VkClearDepthStencilValue clear_value;
    clear_value.depth = clear_value_.depth.depth;
    clear_value.stencil = 0;
    VkImageSubresourceRange subresource_range;
    subresource_range.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    subresource_range.baseMipLevel = 0;
    subresource_range.levelCount = 1;
    subresource_range.baseArrayLayer = 0;
    subresource_range.layerCount = 1;
    vkCmdClearDepthStencilImage(command_buffer, image_->Image()->Handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                &clear_value, 1, &subresource_range);
  }
}

VulkanCmdSetViewport::VulkanCmdSetViewport(const Viewport &viewport) : viewport_(viewport) {
}

void VulkanCmdSetViewport::CompileCommand(VulkanCommandContext *context, VkCommandBuffer command_buffer) {
  VkViewport viewport{};
  viewport.x = viewport_.x;
  viewport.y = viewport_.y + viewport_.height;
  viewport.width = viewport_.width;
  viewport.height = -viewport_.height;
  viewport.minDepth = viewport_.min_depth;
  viewport.maxDepth = viewport_.max_depth;
  vkCmdSetViewport(command_buffer, 0, 1, &viewport);
}

VulkanCmdSetScissor::VulkanCmdSetScissor(const Scissor &scissor) : scissor_(scissor) {
}

void VulkanCmdSetScissor::CompileCommand(VulkanCommandContext *context, VkCommandBuffer command_buffer) {
  VkRect2D scissor{};
  scissor.offset = {scissor_.offset.x, scissor_.offset.y};
  scissor.extent = {scissor_.extent.width, scissor_.extent.height};
  vkCmdSetScissor(command_buffer, 0, 1, &scissor);
}

VulkanCmdSetPrimitiveTopology::VulkanCmdSetPrimitiveTopology(PrimitiveTopology topology) : topology_(topology) {
}

void VulkanCmdSetPrimitiveTopology::CompileCommand(VulkanCommandContext *context, VkCommandBuffer command_buffer) {
  context->Core()->Instance()->Procedures().vkCmdSetPrimitiveTopologyEXT(
      command_buffer, PrimitiveTopologyToVkPrimitiveTopology(topology_));
}

VulkanCmdDraw::VulkanCmdDraw(uint32_t index_count,
                             uint32_t instance_count,
                             int32_t vertex_offset,
                             uint32_t first_instance)
    : index_count_(index_count),
      instance_count_(instance_count),
      vertex_offset_(vertex_offset),
      first_instance_(first_instance) {
}

void VulkanCmdDraw::CompileCommand(VulkanCommandContext *context, VkCommandBuffer command_buffer) {
  vkCmdDraw(command_buffer, index_count_, instance_count_, vertex_offset_, first_instance_);
}

VulkanCmdDrawIndexed::VulkanCmdDrawIndexed(uint32_t index_count,
                                           uint32_t instance_count,
                                           uint32_t first_index,
                                           int32_t vertex_offset,
                                           uint32_t first_instance)
    : index_count_(index_count),
      instance_count_(instance_count),
      first_index_(first_index),
      vertex_offset_(vertex_offset),
      first_instance_(first_instance) {
}

void VulkanCmdDrawIndexed::CompileCommand(VulkanCommandContext *context, VkCommandBuffer command_buffer) {
  vkCmdDrawIndexed(command_buffer, index_count_, instance_count_, first_index_, vertex_offset_, first_instance_);
}

VulkanCmdPresent::VulkanCmdPresent(VulkanWindow *window, VulkanImage *image) : image_(image), window_(window) {
}

void VulkanCmdPresent::CompileCommand(VulkanCommandContext *context, VkCommandBuffer command_buffer) {
  vulkan::TransitImageLayout(command_buffer, window_->CurrentImage(), VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                             VK_IMAGE_ASPECT_COLOR_BIT);

  context->RequireImageState(command_buffer, image_->Image()->Handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT, image_->Image()->Aspect());

  auto image_extent = image_->Extent();
  auto window_extent = window_->SwapChain()->Extent();

  VkImageBlit blit{};
  blit.srcOffsets[0] = {0, 0, 0};
  blit.srcOffsets[1] = {static_cast<int32_t>(image_extent.width), static_cast<int32_t>(image_extent.height), 1};
  blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  blit.srcSubresource.mipLevel = 0;
  blit.srcSubresource.baseArrayLayer = 0;
  blit.srcSubresource.layerCount = 1;
  blit.dstOffsets[0] = {0, 0, 0};
  blit.dstOffsets[1] = {static_cast<int32_t>(window_extent.width), static_cast<int32_t>(window_extent.height), 1};
  blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  blit.dstSubresource.mipLevel = 0;
  blit.dstSubresource.baseArrayLayer = 0;
  blit.dstSubresource.layerCount = 1;
  vkCmdBlitImage(command_buffer, image_->Image()->Handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 window_->CurrentImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

  vulkan::TransitImageLayout(command_buffer, window_->CurrentImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                             VK_IMAGE_ASPECT_COLOR_BIT);
}

VulkanCmdDispatchRays::VulkanCmdDispatchRays(VulkanRayTracingProgram *program,
                                             uint32_t width,
                                             uint32_t height,
                                             uint32_t depth)
    : program_(program), width_(width), height_(height), depth_(depth) {
}

void VulkanCmdDispatchRays::CompileCommand(VulkanCommandContext *context, VkCommandBuffer command_buffer) {
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_pipeline_properties =
      context->Core()->Device()->PhysicalDevice().GetPhysicalDeviceRayTracingPipelineProperties();

  auto aligned_size = [](uint32_t value, uint32_t alignment) { return (value + alignment - 1) & ~(alignment - 1); };
  const uint32_t handle_size_aligned = aligned_size(ray_tracing_pipeline_properties.shaderGroupHandleSize,
                                                    ray_tracing_pipeline_properties.shaderGroupHandleAlignment);

  auto shader_binding_table = program_->ShaderBindingTable();
  VkStridedDeviceAddressRegionKHR ray_gen_shader_sbt_entry{};
  ray_gen_shader_sbt_entry.deviceAddress = shader_binding_table->GetRayGenDeviceAddress();
  ray_gen_shader_sbt_entry.stride = handle_size_aligned;
  ray_gen_shader_sbt_entry.size = handle_size_aligned;

  VkStridedDeviceAddressRegionKHR miss_shader_sbt_entry{};
  miss_shader_sbt_entry.deviceAddress = shader_binding_table->GetMissDeviceAddress();
  miss_shader_sbt_entry.stride = handle_size_aligned;
  miss_shader_sbt_entry.size = handle_size_aligned;

  VkStridedDeviceAddressRegionKHR hit_shader_sbt_entry{};
  hit_shader_sbt_entry.deviceAddress = shader_binding_table->GetClosestHitDeviceAddress();
  hit_shader_sbt_entry.stride = handle_size_aligned;
  hit_shader_sbt_entry.size = handle_size_aligned;

  VkStridedDeviceAddressRegionKHR callable_shader_sbt_entry{};

  program_->Core()->Device()->Procedures().vkCmdTraceRaysKHR(command_buffer, &ray_gen_shader_sbt_entry,
                                                             &miss_shader_sbt_entry, &hit_shader_sbt_entry,
                                                             &callable_shader_sbt_entry, width_, height_, depth_);
}

}  // namespace grassland::graphics::backend
