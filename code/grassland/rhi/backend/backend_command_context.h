#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendCommandContext)
#include <array>
#include <unordered_map>

#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_command_context.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

namespace grassland::rhi::backend {
struct CommandContextState;

class IBackendCommandContext : public rhi::ICommandContext {
 public:
  bool CmdDrawImGui(IWindow *window, IImage *target) override = 0;
  bool CommonCmdDrawImGui(IWindow *window, IImage *target);
  bool CmdPresentImage(IWindow *window, IImage *frame_img, ColorSpace frame_color_space) override = 0;
  bool CommonCmdPresentImage(IWindow *window, IImage *frame_img, ColorSpace frame_color_space);
  bool CmdQueryCompactedSize(IAccelerationStructure *structure) override = 0;
  bool CommonCmdQueryCompactedSize(IAccelerationStructure *structure);
  bool CmdCompactAccelerationStructure(IAccelerationStructure *source,
                                       IAccelerationStructure *destination) override = 0;
  bool CommonCmdCompactAccelerationStructure(IAccelerationStructure *source, IAccelerationStructure *destination);
  bool CmdBuildBottomLevelAccelerationStructure(
      IBottomLevelAccelerationStructure *destination,
      const std::vector<AccelerationStructureGeometryBuffers> &geometries,
      AccelerationStructureBuildMode mode = ACCELERATION_STRUCTURE_BUILD) override = 0;
  bool CommonCmdBuildBottomLevelAccelerationStructure(
      IBottomLevelAccelerationStructure *destination,
      const std::vector<AccelerationStructureGeometryBuffers> &geometries,
      AccelerationStructureBuildMode mode = ACCELERATION_STRUCTURE_BUILD);
  bool CmdBuildTopLevelAccelerationStructure(
      ITopLevelAccelerationStructure *destination,
      const std::vector<RayTracingInstance> &instances,
      AccelerationStructureBuildMode mode = ACCELERATION_STRUCTURE_BUILD) override = 0;
  bool CommonCmdBuildTopLevelAccelerationStructure(ITopLevelAccelerationStructure *destination,
                                                   const std::vector<RayTracingInstance> &instances,
                                                   AccelerationStructureBuildMode mode = ACCELERATION_STRUCTURE_BUILD);
  bool CmdBindRayTracingProgram(IRayTracingProgram *program) override = 0;
  bool CommonCmdBindRayTracingProgram(IRayTracingProgram *program);
  bool CmdDispatchRays(uint32_t width,
                       uint32_t height,
                       uint32_t depth = 1,
                       IShaderBindingTable *sbt = nullptr) override = 0;
  bool CommonCmdDispatchRays(uint32_t width, uint32_t height, uint32_t depth = 1, IShaderBindingTable *sbt = nullptr);

  typedef enum {
    SUBMIT_RESULT_REJECTED,
    SUBMIT_RESULT_SUBMITTED,
    SUBMIT_RESULT_FAILED_IN_FLIGHT,
  } SubmitResult;

 protected:
  void InitializeCommandContext(IBackendQueue *queue);

 public:
  rhi::ISubmissionToken *Submit(const rhi::SubmitInfo &info = {}) override = 0;
  rhi::ISubmissionToken *CommonSubmit(const rhi::SubmitInfo &info = {});

 protected:
  void DestroyCommandContext();

 public:
  bool CmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size) override = 0;
  bool CommonCmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size);
  bool CmdCopyBuffer(IBuffer *source,
                     uint64_t source_offset,
                     IBuffer *destination,
                     uint64_t destination_offset,
                     uint64_t size) override = 0;
  bool CommonCmdCopyBuffer(IBuffer *source,
                           uint64_t source_offset,
                           IBuffer *destination,
                           uint64_t destination_offset,
                           uint64_t size);
  bool CmdBindGroup(uint32_t index, IBindingGroup *group) override = 0;
  bool CommonCmdBindGroup(uint32_t index, IBindingGroup *group);
  bool CmdUploadImage(IImage *image, const void *data, uint64_t row_pitch) override = 0;
  bool CommonCmdUploadImage(IImage *image, const void *data, uint64_t row_pitch);
  bool CmdBindComputeProgram(IComputeProgram *program) override = 0;
  bool CommonCmdBindComputeProgram(IComputeProgram *program);
  bool CmdDispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) override = 0;
  bool CommonCmdDispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1);
  bool CmdBindProgram(IProgram *program) override = 0;
  bool CommonCmdBindProgram(IProgram *program);
  bool CmdBeginRendering(const RenderingInfo &info) override = 0;
  bool CommonCmdBeginRendering(const RenderingInfo &info);
  bool CmdEndRendering() override = 0;
  bool CommonCmdEndRendering();
  bool CmdBindVertexBuffer(uint32_t binding, IBuffer *buffer, uint64_t offset = 0) override = 0;
  bool CommonCmdBindVertexBuffer(uint32_t binding, IBuffer *buffer, uint64_t offset = 0);
  bool CmdBindIndexBuffer(IBuffer *buffer, IndexFormat format = INDEX_FORMAT_UINT32, uint64_t offset = 0) override = 0;
  bool CommonCmdBindIndexBuffer(IBuffer *buffer, IndexFormat format = INDEX_FORMAT_UINT32, uint64_t offset = 0);
  bool CmdSetViewport(const Viewport &viewport) override = 0;
  bool CommonCmdSetViewport(const Viewport &viewport);
  bool CmdSetScissor(const Scissor &scissor) override = 0;
  bool CommonCmdSetScissor(const Scissor &scissor);
  bool CmdDraw(uint32_t vertex_count,
               uint32_t instance_count = 1,
               uint32_t first_vertex = 0,
               uint32_t first_instance = 0) override = 0;
  bool CommonCmdDraw(uint32_t vertex_count,
                     uint32_t instance_count = 1,
                     uint32_t first_vertex = 0,
                     uint32_t first_instance = 0);
  bool CmdDrawIndexed(uint32_t index_count,
                      uint32_t instance_count = 1,
                      uint32_t first_index = 0,
                      int32_t vertex_offset = 0,
                      uint32_t first_instance = 0) override = 0;
  bool CommonCmdDrawIndexed(uint32_t index_count,
                            uint32_t instance_count = 1,
                            uint32_t first_index = 0,
                            int32_t vertex_offset = 0,
                            uint32_t first_instance = 0);
  IDownloadToken *CmdDownloadBuffer(IBuffer *, uint64_t, uint64_t) override = 0;
  IDownloadToken *CommonCmdDownloadBuffer(IBuffer *, uint64_t, uint64_t);
  IDownloadToken *CmdDownloadImage(IImage *, const ImageDownloadRegion & = {}) override = 0;
  IDownloadToken *CommonCmdDownloadImage(IImage *, const ImageDownloadRegion & = {});
  void BindDownloads(IBackendSubmissionToken *submission);
  void ReleaseDownloads();
  bool Seal();
  const void *DeviceIdentity() const;
  IBackendQueue *OwnerQueue() const;
  void MarkSubmitted();
  bool WasSubmitted() const;
  virtual void UpdatePresentedImages() = 0;
  virtual SubmitResult SubmitNative(const SubmitInfo &info, Event signal) = 0;
  virtual SubmissionStatus PollCompletion(Event signal, uint64_t completed, bool timeline_valid) = 0;

 protected:
  // Call before encoding each resource use. Retains each exact object once.
  IDownloadToken *RegisterDownload(ObjectPtr<IBackendDownloadToken> token);
  bool RetainResource(rhi::IObject *resource);

  template <typename T, std::enable_if_t<!std::is_convertible_v<T *, IObject *>, int> = 0>
  bool RetainResource(T *resource) {
    return RetainResource(rhi::QueryInterfacePtr<IObject>(resource).Get());
  }

  virtual bool EndRecording() = 0;
  bool PrepareRendering(const RenderingInfo &info);
  bool RenderingMatches(const IProgram *program) const;
  virtual CommandContextState &StateCommandContext() = 0;
  virtual const CommandContextState &StateCommandContext() const = 0;
};

}  // namespace grassland::rhi::backend
