#pragma once
#include <array>
#include <unordered_map>

#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_command_context.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendCommandContext)

namespace grassland::rhi::backend {
class IBackendCommandContext : public rhi::ICommandContext {
 public:
  bool CmdDrawImGui(IWindow *window, IImage *target) override = 0;

  bool CmdPresentImage(IWindow *window, IImage *frame_img, ColorSpace frame_color_space) override = 0;

  bool CmdQueryCompactedSize(IAccelerationStructure *structure) override = 0;

  bool CmdCompactAccelerationStructure(IAccelerationStructure *source,
                                       IAccelerationStructure *destination) override = 0;

  bool CmdBuildBottomLevelAccelerationStructure(
      IBottomLevelAccelerationStructure *destination,
      const std::vector<AccelerationStructureGeometryBuffers> &geometries,
      AccelerationStructureBuildMode mode = ACCELERATION_STRUCTURE_BUILD) override = 0;

  bool CmdBuildTopLevelAccelerationStructure(
      ITopLevelAccelerationStructure *destination,
      const std::vector<RayTracingInstance> &instances,
      AccelerationStructureBuildMode mode = ACCELERATION_STRUCTURE_BUILD) override = 0;

  bool CmdBindRayTracingProgram(IRayTracingProgram *program) override = 0;

  bool CmdDispatchRays(uint32_t width,
                       uint32_t height,
                       uint32_t depth = 1,
                       IShaderBindingTable *sbt = nullptr) override = 0;

  typedef enum {
    SUBMIT_RESULT_REJECTED,
    SUBMIT_RESULT_SUBMITTED,
    SUBMIT_RESULT_FAILED_IN_FLIGHT,
  } SubmitResult;

  rhi::ISubmissionToken *Submit(const rhi::SubmitInfo &info = {}) override = 0;

 protected:
  void DestroyCommandContext();

 public:
  bool CmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size) override = 0;

  bool CmdCopyBuffer(IBuffer *source,
                     uint64_t source_offset,
                     IBuffer *destination,
                     uint64_t destination_offset,
                     uint64_t size) override = 0;

  bool CmdBindGroup(uint32_t index, IBindingGroup *group) override = 0;

  bool CmdUploadImage(IImage *image, const void *data, uint64_t row_pitch) override = 0;

  bool CmdBindComputeProgram(IComputeProgram *program) override = 0;

  bool CmdDispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) override = 0;

  bool CmdBindProgram(IProgram *program) override = 0;

  bool CmdBeginRendering(const RenderingInfo &info) override = 0;

  bool CmdEndRendering() override = 0;

  bool CmdBindVertexBuffer(uint32_t binding, IBuffer *buffer, uint64_t offset = 0) override = 0;

  bool CmdBindIndexBuffer(IBuffer *buffer, IndexFormat format = INDEX_FORMAT_UINT32, uint64_t offset = 0) override = 0;

  bool CmdSetViewport(const Viewport &viewport) override = 0;

  bool CmdSetScissor(const Scissor &scissor) override = 0;

  bool CmdDraw(uint32_t vertex_count,
               uint32_t instance_count = 1,
               uint32_t first_vertex = 0,
               uint32_t first_instance = 0) override = 0;

  bool CmdDrawIndexed(uint32_t index_count,
                      uint32_t instance_count = 1,
                      uint32_t first_index = 0,
                      int32_t vertex_offset = 0,
                      uint32_t first_instance = 0) override = 0;

  IDownloadToken *CmdDownloadBuffer(IBuffer *, uint64_t, uint64_t) override = 0;

  IDownloadToken *CmdDownloadImage(IImage *, const ImageDownloadRegion & = {}) override = 0;

  virtual void BindDownloads(IBackendSubmissionToken *submission) = 0;

  virtual void ReleaseDownloads() = 0;

  virtual bool Seal() = 0;

  virtual const void *DeviceIdentity() const = 0;

  virtual IBackendQueue *OwnerQueue() const = 0;

  virtual void MarkSubmitted() = 0;

  virtual bool WasSubmitted() const = 0;

  virtual void UpdatePresentedImages() = 0;

  virtual SubmitResult SubmitNative(const SubmitInfo &info, Event signal) = 0;

  virtual SubmissionStatus PollCompletion(Event signal, uint64_t completed, bool timeline_valid) = 0;

 protected:
  virtual bool EndRecording() = 0;
};

}  // namespace grassland::rhi::backend
