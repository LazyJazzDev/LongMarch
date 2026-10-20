#pragma once
#include <vector>

#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"

namespace grassland::rhi::backend::vulkan {
class VulkanImage;
class VulkanBuffer;
class VulkanProgram;
class VulkanWindow;

class VulkanCommandContext final : public CommandContext {
 public:
  ~VulkanCommandContext() override;
  static VulkanCommandContext *Create(VulkanDevice *device);
  bool CmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size) override;
  bool CmdCopyBuffer(IBuffer *source,
                     uint64_t source_offset,
                     IBuffer *destination,
                     uint64_t destination_offset,
                     uint64_t size) override;
  bool CmdBindGroup(uint32_t index, IBindingGroup *group) override;
  bool CmdUploadImage(IImage *image, const void *data, uint64_t row_pitch) override;
  bool CmdBindProgram(IProgram *program) override;
  bool CmdBeginRendering(const RenderingInfo &info) override;
  bool CmdEndRendering() override;
  bool CmdBindVertexBuffer(uint32_t binding, IBuffer *buffer, uint64_t offset = 0) override;
  bool CmdBindIndexBuffer(IBuffer *buffer, IndexFormat format = INDEX_FORMAT_UINT32, uint64_t offset = 0) override;
  bool CmdSetViewport(const Viewport &viewport) override;
  bool CmdSetScissor(const Scissor &scissor) override;
  bool CmdDraw(uint32_t vertex_count,
               uint32_t instance_count = 1,
               uint32_t first_vertex = 0,
               uint32_t first_instance = 0) override;
  bool CmdDrawIndexed(uint32_t index_count,
                      uint32_t instance_count = 1,
                      uint32_t first_index = 0,
                      int32_t vertex_offset = 0,
                      uint32_t first_instance = 0) override;
  bool CmdClearColor(rhi::IImage *image, const std::array<float, 4> &color) override;
  bool CmdPresent(rhi::IWindow *window) override;
  SubmitResult SubmitNative() override;
  SubmissionStatus WaitForProgress() override;

 private:
  VulkanCommandContext(VulkanDevice *device);

  bool ValidateGroups();
  std::vector<VulkanBindingGroup *> groups_;
  VulkanProgram *program_{};
  VulkanImage *render_target_{};
  std::vector<VulkanBuffer *> vertex_buffers_;
  bool viewport_set_{}, scissor_set_{}, index_bound_{};

  bool EndRecording() override;
  bool TransitionBuffer(VulkanBuffer *buffer, VkPipelineStageFlags stages, VkAccessFlags access);

  struct BufferState {
    VkPipelineStageFlags stages;
    VkAccessFlags access;
  };

  std::unordered_map<VulkanBuffer *, BufferState> buffer_states_;
  bool TransitionImage(VulkanImage *image, VkImageLayout layout, VkPipelineStageFlags stages, VkAccessFlags access);

  struct ImageState {
    VkImageLayout layout;
    VkPipelineStageFlags stages;
    VkAccessFlags access;
  };

  std::unordered_map<VulkanImage *, ImageState> image_states_;
  std::vector<VulkanWindow *> presents_;
  VulkanDevice *device_;
  VkCommandPool pool_{};
  VkCommandBuffer command_{};
  uint64_t value_{};
  bool submission_failed_{};
  bool reusable_{};
};

}  // namespace grassland::rhi::backend::vulkan
