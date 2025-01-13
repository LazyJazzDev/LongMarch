#include "grassland/graphics/backend/vulkan/vulkan_command_context.h"

#include "grassland/graphics/backend/vulkan/vulkan_acceleration_structure.h"
#include "grassland/graphics/backend/vulkan/vulkan_buffer.h"
#include "grassland/graphics/backend/vulkan/vulkan_image.h"
#include "grassland/graphics/backend/vulkan/vulkan_program.h"
#include "grassland/graphics/backend/vulkan/vulkan_sampler.h"
#include "grassland/graphics/backend/vulkan/vulkan_window.h"

namespace grassland::graphics::backend {

VulkanCommandContext::VulkanCommandContext(VulkanCore *core) : core_(core) {
  for (int i = 0; i < BIND_POINT_COUNT; i++) {
    program_bases_[i] = nullptr;
  }
}

void VulkanCommandContext::CmdBindProgram(Program *program) {
  auto vk_program = dynamic_cast<VulkanProgram *>(program);
  assert(vk_program != nullptr);
  program_bases_[BIND_POINT_GRAPHICS] = vk_program;
  commands_.push_back(std::make_unique<VulkanCmdBindProgram>(vk_program));
}

void VulkanCommandContext::CmdBindRayTracingProgram(RayTracingProgram *program) {
  VulkanRayTracingProgram *vk_program = dynamic_cast<VulkanRayTracingProgram *>(program);
  assert(vk_program != nullptr);
  program_bases_[BIND_POINT_RAYTRACING] = vk_program;
  commands_.push_back(std::make_unique<VulkanCmdBindRayTracingProgram>(vk_program));
}

void VulkanCommandContext::CmdBindVertexBuffers(uint32_t first_binding,
                                                const std::vector<Buffer *> &buffers,
                                                const std::vector<uint64_t> &offsets) {
  std::vector<VulkanBuffer *> vertex_buffers(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    vertex_buffers[i] = dynamic_cast<VulkanBuffer *>(buffers[i]);
    RecordDynamicBuffer(vertex_buffers[i]);
  }
  commands_.push_back(std::make_unique<VulkanCmdBindVertexBuffers>(first_binding, vertex_buffers, offsets));
}

void VulkanCommandContext::CmdBindIndexBuffer(Buffer *buffer, uint64_t offset) {
  auto index_buffer = dynamic_cast<VulkanBuffer *>(buffer);
  RecordDynamicBuffer(index_buffer);
  commands_.push_back(std::make_unique<VulkanCmdBindIndexBuffer>(index_buffer, offset));
}

void VulkanCommandContext::CmdBeginRendering(const std::vector<Image *> &color_targets, Image *depth_target) {
  std::vector<VulkanImage *> vk_color_targets(color_targets.size());
  VulkanImage *vk_depth_target{nullptr};

  for (size_t i = 0; i < color_targets.size(); ++i) {
    vk_color_targets[i] = dynamic_cast<VulkanImage *>(color_targets[i]);
  }

  if (depth_target) {
    vk_depth_target = dynamic_cast<VulkanImage *>(depth_target);
  }

  auto cmd = std::make_unique<VulkanCmdBeginRendering>(vk_color_targets, vk_depth_target);

  active_rendering_cmd_ = cmd.get();

  commands_.push_back(std::move(cmd));
}

void VulkanCommandContext::CmdBindResources(int slot, const std::vector<Buffer *> &buffers, BindPoint bind_point) {
  if (!program_bases_[bind_point]) {
    LogError("[Graphics.Vulkan] Program on bind point {} is not set", int(bind_point));
    return;
  }
  std::vector<VulkanBuffer *> vk_buffers(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    vk_buffers[i] = dynamic_cast<VulkanBuffer *>(buffers[i]);
    RecordDynamicBuffer(vk_buffers[i]);
  }
  commands_.push_back(
      std::make_unique<VulkanCmdBindResourceBuffers>(slot, vk_buffers, program_bases_[bind_point], bind_point));
  required_pool_size_ += program_bases_[bind_point]->DescriptorSetLayout(slot)->GetPoolSize();
  required_set_count_++;
}

void VulkanCommandContext::CmdBindResources(int slot, const std::vector<Image *> &images, BindPoint bind_point) {
  if (!program_bases_[bind_point]) {
    LogError("[Graphics.Vulkan] Program on bind point {} is not set", int(bind_point));
    return;
  }
  std::vector<VulkanImage *> vk_images(images.size());
  bool update_layout = true;
  for (size_t i = 0; i < images.size(); ++i) {
    vk_images[i] = dynamic_cast<VulkanImage *>(images[i]);
    assert(vk_images[i] != nullptr);
    if (bind_point == BIND_POINT_GRAPHICS && active_rendering_cmd_) {
      active_rendering_cmd_->RecordResourceImages(vk_images[i]);
      update_layout = false;
    }
  }
  commands_.push_back(std::make_unique<VulkanCmdBindResourceImages>(slot, vk_images, program_bases_[bind_point],
                                                                    bind_point, update_layout));
  required_pool_size_ += program_bases_[bind_point]->DescriptorSetLayout(slot)->GetPoolSize();
  required_set_count_++;
}

void VulkanCommandContext::CmdBindResources(int slot, const std::vector<Sampler *> &samplers, BindPoint bind_point) {
  if (!program_bases_[bind_point]) {
    LogError("[Graphics.Vulkan] Program on bind point {} is not set", int(bind_point));
    return;
  }
  std::vector<VulkanSampler *> vk_samplers(samplers.size());
  for (size_t i = 0; i < samplers.size(); ++i) {
    vk_samplers[i] = dynamic_cast<VulkanSampler *>(samplers[i]);
  }
  commands_.push_back(
      std::make_unique<VulkanCmdBindResourceSamplers>(slot, vk_samplers, program_bases_[bind_point], bind_point));
  required_pool_size_ += program_bases_[bind_point]->DescriptorSetLayout(slot)->GetPoolSize();
  required_set_count_++;
}

void VulkanCommandContext::CmdBindResources(int slot,
                                            AccelerationStructure *acceleration_structure,
                                            BindPoint bind_point) {
  if (!program_bases_[bind_point]) {
    LogError("[Graphics.Vulkan] Program on bind point {} is not set", int(bind_point));
    return;
  }
  auto vk_acceleration_structure = dynamic_cast<VulkanAccelerationStructure *>(acceleration_structure);
  commands_.push_back(std::make_unique<VulkanCmdBindResourceAccelerationStructure>(
      slot, vk_acceleration_structure, program_bases_[bind_point], bind_point));
  required_pool_size_ += program_bases_[bind_point]->DescriptorSetLayout(slot)->GetPoolSize();
  required_set_count_++;
}

void VulkanCommandContext::CmdEndRendering() {
  active_rendering_cmd_ = nullptr;
  commands_.push_back(std::make_unique<VulkanCmdEndRendering>());
}

void VulkanCommandContext::CmdSetViewport(const Viewport &viewport) {
  commands_.push_back(std::make_unique<VulkanCmdSetViewport>(viewport));
}

void VulkanCommandContext::CmdSetScissor(const Scissor &scissor) {
  commands_.push_back(std::make_unique<VulkanCmdSetScissor>(scissor));
}

void VulkanCommandContext::CmdSetPrimitiveTopology(PrimitiveTopology topology) {
  commands_.push_back(std::make_unique<VulkanCmdSetPrimitiveTopology>(topology));
}

void VulkanCommandContext::CmdDrawIndexed(uint32_t index_count,
                                          uint32_t instance_count,
                                          uint32_t first_index,
                                          int32_t vertex_offset,
                                          uint32_t first_instance) {
  commands_.push_back(
      std::make_unique<VulkanCmdDrawIndexed>(index_count, instance_count, first_index, vertex_offset, first_instance));
}

void VulkanCommandContext::CmdClearImage(Image *image, const ClearValue &color) {
  commands_.push_back(std::make_unique<VulkanCmdClearImage>(dynamic_cast<VulkanImage *>(image), color));
}

void VulkanCommandContext::CmdPresent(Window *window, Image *image) {
  auto vulkan_window = dynamic_cast<VulkanWindow *>(window);
  auto vulkan_image = dynamic_cast<VulkanImage *>(image);
  commands_.push_back(std::make_unique<VulkanCmdPresent>(vulkan_window, vulkan_image));
  windows_.insert(vulkan_window);
}

void VulkanCommandContext::CmdDispatchRays(uint32_t width, uint32_t height, uint32_t depth) {
  commands_.push_back(std::make_unique<VulkanCmdDispatchRays>(
      dynamic_cast<VulkanRayTracingProgram *>(program_bases_[BIND_POINT_RAYTRACING]), width, height, depth));
}

void VulkanCommandContext::RequireImageState(VkCommandBuffer cmd_buffer,
                                             VkImage image,
                                             VkImageLayout layout,
                                             VkPipelineStageFlags stage,
                                             VkAccessFlags access,
                                             VkImageAspectFlags aspect) {
  if (image_states_.count(image) == 0) {
    image_states_[image] = {VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_ACCESS_NONE};
  }
  auto &state = image_states_[image];
  vulkan::TransitImageLayout(cmd_buffer, image, state.layout, layout, state.stage, stage, state.access, access, aspect);
  state = {layout, stage, access, aspect};
}

void VulkanCommandContext::RecordDynamicBuffer(VulkanBuffer *buffer) {
  auto dynamic_buffer = dynamic_cast<VulkanDynamicBuffer *>(buffer);
  if (dynamic_buffer) {
    dynamic_buffers_.insert(dynamic_buffer);
  }
}

}  // namespace grassland::graphics::backend
