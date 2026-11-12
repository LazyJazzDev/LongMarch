#pragma once
#include "grassland/rhi/backend/metal/metal_command_context.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
class MetalCommandContextImpl final : public Object<rhi::ICommandContext, IBackendCommandContext> {
 public:
  static ICommandContext *Create(IMetalQueue *queue);
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
  // Inputs and ranges have been validated by the recording entry point.
  bool CopyBuffer(IMetalBuffer *source,
                  uint64_t source_offset,
                  IMetalBuffer *destination,
                  uint64_t destination_offset,
                  uint64_t size);
  MetalCommandContextImpl(IMetalQueue *queue);
  bool EndRecording() override;
  std::vector<MTL::Resource *> AccelerationResources();
  std::unordered_map<IMetalTopLevelAccelerationStructure *, std::vector<ObjectPtr<IBottomLevelAccelerationStructure>>>
      as_dependencies_;
  void ApplyBindings();
  void ApplyScissor();
  bool ValidateDraw(bool indexed);
  IProgram *program_{};
  IMetalComputeProgram *compute_program_{};
  IMetalProgramBase *native_program_{};
  IImage *render_target_{};
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

 public:
  bool CmdDrawImGui(IWindow *window, IImage *target) override;

  bool CmdPresentImage(IWindow *window, IImage *frame_img, ColorSpace frame_color_space) override;

  bool CmdBindRayTracingProgram(IRayTracingProgram *program) override;

  bool CmdDispatchRays(uint32_t width,
                       uint32_t height,
                       uint32_t depth = 1,
                       IShaderBindingTable *sbt = nullptr) override;

 private:
  bool PrepareRendering(const RenderingInfo &info);

  bool RenderingMatches(const IProgram *program) const;

  void InitializeCommandContext(IMetalQueue *queue);

 public:
  rhi::ISubmissionToken *Submit(const rhi::SubmitInfo &info = {}) override;

  void BindDownloads(rhi::ISubmissionToken *submission) override;

  void ReleaseDownloads() override;

 private:
  // Call before encoding each resource use. Retains each exact object once.
  IDownloadToken *RegisterDownload(ObjectPtr<IMetalDownloadToken> token);

  template <typename T, std::enable_if_t<!std::is_convertible_v<T *, rhi::IObject *>, int> = 0>
  bool RetainResource(T *resource) {
    return RetainResource(rhi::QueryInterfacePtr<rhi::IObject>(resource).Get());
  }

  bool RetainResource(rhi::IObject *resource);

 public:
  bool Seal() override;

  IBackendQueue *OwnerQueue() const override;

  const void *DeviceIdentity() const override;

  void MarkSubmitted() override;

  bool WasSubmitted() const override;

 private:
  RenderingInfo rendering_{};
  bool recording_failed_{};
  bool sealed_{};
  IMetalQueue *queue_{};
  std::unordered_map<rhi::IObject *, ObjectPtr<rhi::IObject>> resources_{};
  bool submitted_{};
  std::vector<ObjectPtr<IMetalDownloadToken>> downloads_{};
};
}  // namespace grassland::rhi::backend::metal
