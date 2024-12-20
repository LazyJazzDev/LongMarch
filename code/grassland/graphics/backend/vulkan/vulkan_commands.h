#pragma once
#include "grassland/graphics/backend/vulkan/vulkan_core.h"

namespace grassland::graphics::backend {
class VulkanCommand {
 public:
  virtual ~VulkanCommand() = default;
  virtual void CompileCommand(VulkanCommandContext *context,
                              VkCommandBuffer command_buffer) = 0;
};

class VulkanCmdBindProgram : public VulkanCommand {
 public:
  VulkanCmdBindProgram(VulkanProgram *program);

  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;

 private:
  VulkanProgram *program_;
};

class VulkanCmdBindVertexBuffers : public VulkanCommand {
 public:
  VulkanCmdBindVertexBuffers(uint32_t first_binding,
                             const std::vector<VulkanBuffer *> &buffers,
                             const std::vector<uint64_t> &offsets);

  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;

 private:
  uint32_t first_binding_;
  std::vector<VulkanBuffer *> buffers_;
  std::vector<VkDeviceSize> offsets_;
};

class VulkanCmdBindIndexBuffer : public VulkanCommand {
 public:
  VulkanCmdBindIndexBuffer(VulkanBuffer *buffer, uint64_t offset);

  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;

 private:
  VulkanBuffer *buffer_;
  VkDeviceSize offset_;
};

class VulkanCmdBeginRendering : public VulkanCommand {
 public:
  VulkanCmdBeginRendering(const std::vector<VulkanImage *> &color_targets,
                          VulkanImage *depth_target);

  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;

  void RecordResourceImages(VulkanImage *resource_image);

 private:
  std::vector<VulkanImage *> color_targets_;
  VulkanImage *depth_target_;
  std::set<VulkanImage *> resource_images_;
};

class VulkanCmdBindResourceBuffers : public VulkanCommand {
 public:
  VulkanCmdBindResourceBuffers(int slot,
                               const std::vector<VulkanBuffer *> &buffers,
                               VulkanProgram *program);

  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;

 private:
  int slot_;
  std::vector<VulkanBuffer *> buffers_;
  VulkanProgram *program_;
};

class VulkanCmdBindResourceImages : public VulkanCommand {
 public:
  VulkanCmdBindResourceImages(int slot,
                              const std::vector<VulkanImage *> &images,
                              VulkanProgram *program);

  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;

 private:
  int slot_;
  std::vector<VulkanImage *> images_;
  VulkanProgram *program_;
};

class VulkanCmdBindResourceSamplers : public VulkanCommand {
 public:
  VulkanCmdBindResourceSamplers(int slot,
                                const std::vector<VulkanSampler *> &samplers,
                                VulkanProgram *program);

  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;

 private:
  int slot_;
  std::vector<VulkanSampler *> samplers_;
  VulkanProgram *program_;
};

class VulkanCmdEndRendering : public VulkanCommand {
 public:
  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;
};

class VulkanCmdClearImage : public VulkanCommand {
 public:
  VulkanCmdClearImage(VulkanImage *image, const ClearValue &clear_value);

  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;

 private:
  VulkanImage *image_;
  ClearValue clear_value_;
};

class VulkanCmdSetViewport : public VulkanCommand {
 public:
  VulkanCmdSetViewport(const Viewport &viewport);

  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;

 private:
  Viewport viewport_;
};

class VulkanCmdSetScissor : public VulkanCommand {
 public:
  VulkanCmdSetScissor(const Scissor &scissor);

  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;

 private:
  Scissor scissor_;
};

class VulkanCmdSetPrimitiveTopology : public VulkanCommand {
 public:
  VulkanCmdSetPrimitiveTopology(PrimitiveTopology topology);

  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;

 private:
  PrimitiveTopology topology_;
};

class VulkanCmdDrawIndexed : public VulkanCommand {
 public:
  VulkanCmdDrawIndexed(uint32_t index_count,
                       uint32_t instance_count,
                       uint32_t first_index,
                       int32_t vertex_offset,
                       uint32_t first_instance);

  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;

 private:
  uint32_t index_count_;
  uint32_t instance_count_;
  uint32_t first_index_;
  int32_t vertex_offset_;
  uint32_t first_instance_;
};

class VulkanCmdPresent : public VulkanCommand {
 public:
  VulkanCmdPresent(VulkanWindow *window, VulkanImage *image);
  void CompileCommand(VulkanCommandContext *context,
                      VkCommandBuffer command_buffer) override;

 private:
  VulkanImage *image_;
  VulkanWindow *window_;
};

}  // namespace grassland::graphics::backend
