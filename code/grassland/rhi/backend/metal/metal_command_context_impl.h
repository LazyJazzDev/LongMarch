#pragma once
#include "grassland/rhi/backend/backend_command_context_state.h"
#include "grassland/rhi/backend/metal/metal_command_context.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
class MetalCommandContextImpl final : public Object<IMetalCommandContext> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::ICommandContext, IMetalCommandContext, IBackendCommandContext, rhi::ICommandContext)
  static IMetalCommandContext *Create(IMetalQueue *queue);
  ~MetalCommandContextImpl() override;
  bool CmdBuildBottomLevelAccelerationStructure(
      IBottomLevelAccelerationStructure *destination,
      const std::vector<AccelerationStructureGeometryBuffers> &geometries,
      AccelerationStructureBuildMode mode = ACCELERATION_STRUCTURE_BUILD) override;
  bool CmdBuildTopLevelAccelerationStructure(
      ITopLevelAccelerationStructure *destination,
      const std::vector<RayTracingInstance> &instances,
      AccelerationStructureBuildMode mode = ACCELERATION_STRUCTURE_BUILD) override;
  bool CmdQueryCompactedSize(IAccelerationStructure *structure) override;
  bool CmdCompactAccelerationStructure(IAccelerationStructure *source, IAccelerationStructure *destination) override;
  IDownloadToken *CmdDownloadBuffer(IBuffer *buffer, uint64_t offset, uint64_t size) override;
  IDownloadToken *CmdDownloadImage(IImage *image, const ImageDownloadRegion &region = {}) override;
  bool CmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size) override;
  bool CmdCopyBuffer(IBuffer *source,
                     uint64_t source_offset,
                     IBuffer *destination,
                     uint64_t destination_offset,
                     uint64_t size) override;
  bool CmdBindComputeProgram(IComputeProgram *program) override;
  bool CmdDispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) override;
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
  bool CmdClearImage(rhi::IImage *image, const ClearValue &value) override;
  bool CmdPresent(rhi::IWindow *window) override;
  void UpdatePresentedImages() override;
  SubmitResult SubmitNative(const SubmitInfo &info, Event signal) override;
  SubmissionStatus PollCompletion(Event signal, uint64_t completed, bool timeline_valid) override;

 private:
  MetalCommandContextImpl(IMetalQueue *queue);
  bool EndRecording() override;
  std::vector<MTL::Resource *> AccelerationResources();
  std::unordered_map<IMetalTopLevelAccelerationStructure *, std::vector<ObjectPtr<IBottomLevelAccelerationStructure>>>
      as_dependencies_;
  void ApplyBindings();
  void ApplyScissor();
  bool ValidateDraw(bool indexed);
  IMetalProgram *program_{};
  IMetalComputeProgram *compute_program_{};
  IMetalProgramBase *native_program_{};
  IMetalImage *render_target_{};
  std::vector<ImageFormat> color_formats_;
  std::vector<IImage *> color_targets_;
  ImageFormat depth_format_{IMAGE_FORMAT_UNKNOWN};
  std::vector<IMetalBindingVersion *> groups_;
  NS::SharedPtr<MTL::RenderCommandEncoder> encoder_;
  std::vector<std::pair<IMetalBuffer *, uint64_t>> vertex_buffers_;
  IMetalBuffer *index_{};
  uint64_t index_offset_{};
  IndexFormat index_format_{INDEX_FORMAT_UINT32};
  Viewport viewport_{};
  Scissor scissor_{};
  bool viewport_set_{}, scissor_set_{};
  IMetalDevice *device_;
  NS::SharedPtr<MTL::CommandBuffer> commands_;
  std::vector<std::pair<IMetalWindow *, IMetalSwapChainImage *>> presents_;

 private:
  CommandContextState state_CommandContext_;

  CommandContextState &StateCommandContext() override {
    return state_CommandContext_;
  }

  const CommandContextState &StateCommandContext() const override {
    return state_CommandContext_;
  }

 public:
  bool CmdDrawImGui(IWindow *window, IImage *target) override {
    return IBackendCommandContext::CommonCmdDrawImGui(window, target);
  }

  bool CmdPresentImage(IWindow *window, IImage *frame_img, ColorSpace frame_color_space) override {
    return IBackendCommandContext::CommonCmdPresentImage(window, frame_img, frame_color_space);
  }

  bool CmdBindRayTracingProgram(IRayTracingProgram *program) override {
    return IBackendCommandContext::CommonCmdBindRayTracingProgram(program);
  }

  bool CmdDispatchRays(uint32_t width,
                       uint32_t height,
                       uint32_t depth = 1,
                       IShaderBindingTable *sbt = nullptr) override {
    return IBackendCommandContext::CommonCmdDispatchRays(width, height, depth, sbt);
  }

  rhi::ISubmissionToken *Submit(const rhi::SubmitInfo &info = {}) override {
    return IBackendCommandContext::CommonSubmit(info);
  }
};
}  // namespace grassland::rhi::backend::metal
