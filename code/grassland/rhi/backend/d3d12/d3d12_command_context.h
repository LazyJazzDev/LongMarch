#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12CommandContext)
#include <vector>

#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/d3d12/d3d12_binding_group.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12CommandContext : public IBackendCommandContext {
 public:
  static ID3D12CommandContext *Create(ID3D12Queue *queue);
  virtual bool CmdBindRayTracingProgram(IRayTracingProgram *program) = 0;
  virtual bool CmdDispatchRays(uint32_t width,
                               uint32_t height,
                               uint32_t depth = 1,
                               IShaderBindingTable *sbt = nullptr) = 0;
  virtual bool CmdQueryCompactedSize(IAccelerationStructure *structure) = 0;
  virtual bool CmdCompactAccelerationStructure(IAccelerationStructure *source, IAccelerationStructure *destination) = 0;
  virtual bool CmdBuildBottomLevelAccelerationStructure(
      IBottomLevelAccelerationStructure *destination,
      const std::vector<AccelerationStructureGeometryBuffers> &geometries,
      AccelerationStructureBuildMode mode = ACCELERATION_STRUCTURE_BUILD) = 0;
  virtual bool CmdBuildTopLevelAccelerationStructure(
      ITopLevelAccelerationStructure *destination,
      const std::vector<RayTracingInstance> &instances,
      AccelerationStructureBuildMode mode = ACCELERATION_STRUCTURE_BUILD) = 0;
  virtual IDownloadToken *CmdDownloadBuffer(IBuffer *buffer, uint64_t offset, uint64_t size) = 0;
  virtual IDownloadToken *CmdDownloadImage(IImage *image, const ImageDownloadRegion &region = {}) = 0;
  virtual bool CmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size) = 0;
  virtual bool CmdCopyBuffer(IBuffer *source,
                             uint64_t source_offset,
                             IBuffer *destination,
                             uint64_t destination_offset,
                             uint64_t size) = 0;
  virtual bool CmdBindGroup(uint32_t index, IBindingGroup *group) = 0;
  virtual bool CmdUploadImage(IImage *image, const void *data, uint64_t row_pitch) = 0;
  virtual bool CmdBindComputeProgram(IComputeProgram *program) = 0;
  virtual bool CmdDispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) = 0;
  virtual bool CmdBindProgram(IProgram *program) = 0;
  virtual bool CmdBeginRendering(const RenderingInfo &info) = 0;
  virtual bool CmdEndRendering() = 0;
  virtual bool CmdBindVertexBuffer(uint32_t binding, IBuffer *buffer, uint64_t offset = 0) = 0;
  virtual bool CmdBindIndexBuffer(IBuffer *buffer, IndexFormat format = INDEX_FORMAT_UINT32, uint64_t offset = 0) = 0;
  virtual bool CmdSetViewport(const Viewport &viewport) = 0;
  virtual bool CmdSetScissor(const Scissor &scissor) = 0;
  virtual bool CmdDraw(uint32_t vertex_count,
                       uint32_t instance_count = 1,
                       uint32_t first_vertex = 0,
                       uint32_t first_instance = 0) = 0;
  virtual bool CmdDrawIndexed(uint32_t index_count,
                              uint32_t instance_count = 1,
                              uint32_t first_index = 0,
                              int32_t vertex_offset = 0,
                              uint32_t first_instance = 0) = 0;
  virtual bool CmdClearImage(rhi::IImage *image, const ClearValue &value) = 0;
  virtual bool CmdPresent(rhi::IWindow *window) = 0;
  virtual void UpdatePresentedImages() = 0;
  virtual SubmitResult SubmitNative(const SubmitInfo &info, Event signal) = 0;
  virtual SubmissionStatus PollCompletion(Event signal, uint64_t completed, bool timeline_valid) = 0;
};

;

}  // namespace grassland::rhi::backend::d3d12
