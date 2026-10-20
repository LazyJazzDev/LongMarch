#pragma once
#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/metal/metal_binding_group.h"
#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {
class MetalProgram;
class MetalBuffer;
class MetalImage;
class MetalWindow;
class MetalSwapChainImage;

class MetalCommandContext final : public CommandContext {
 public:
  static MetalCommandContext *Create(MetalDevice *device);
  ~MetalCommandContext() override;
  bool CmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size) override;
  bool CmdCopyBuffer(IBuffer *source,
                     uint64_t source_offset,
                     IBuffer *destination,
                     uint64_t destination_offset,
                     uint64_t size) override;
  bool CmdBindProgram(IProgram *program) override;
  bool CmdBindGroup(uint32_t index, IBindingGroup *group) override;
  bool CmdUploadImage(IImage *image, const void *data, uint64_t row_pitch) override;
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
  MetalCommandContext(MetalDevice *device);

  bool EndRecording() override;
  void ApplyBindings();
  void ApplyScissor();
  bool ValidateDraw(bool indexed);
  MetalProgram *program_{};
  MetalImage *render_target_{};
  std::vector<ImageFormat> color_formats_;
  std::vector<IImage *> color_targets_;
  ImageFormat depth_format_{IMAGE_FORMAT_UNKNOWN};
  std::vector<MetalBindingGroup::Version *> groups_;
  NS::SharedPtr<MTL::RenderCommandEncoder> encoder_;
  std::vector<std::pair<MetalBuffer *, uint64_t>> vertex_buffers_;
  MetalBuffer *index_{};
  uint64_t index_offset_{};
  IndexFormat index_format_{INDEX_FORMAT_UINT32};
  Viewport viewport_{};
  Scissor scissor_{};
  bool viewport_set_{}, scissor_set_{};

  MetalDevice *device_;
  NS::SharedPtr<MTL::CommandBuffer> commands_;
  std::vector<std::pair<MetalWindow *, MetalSwapChainImage *>> presents_;
  bool submission_failed_{};
};

}  // namespace grassland::rhi::backend::metal
