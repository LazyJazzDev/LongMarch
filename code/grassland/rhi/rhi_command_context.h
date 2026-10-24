#pragma once
#include <array>

#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_download_token.h"
#include "grassland/rhi/rhi_object.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"
#include "grassland/rhi/rhi_util.h"

namespace grassland::rhi {
class IBuffer;
class IProgram;
class IComputeProgram;
class IBindingGroup;
class IImage;

struct ColorAttachmentInfo {
  IImage *color_attachment{};
  AttachmentLoadOp load_op{ATTACHMENT_LOAD_OP_LOAD};
  AttachmentStoreOp store_op{ATTACHMENT_STORE_OP_STORE};
  ClearColorValue clear_value{};
};

struct DepthStencilAttachmentInfo {
  IImage *image{};
  AttachmentLoadOp depth_load_op{ATTACHMENT_LOAD_OP_LOAD};
  AttachmentStoreOp depth_store_op{ATTACHMENT_STORE_OP_STORE};
  AttachmentLoadOp stencil_load_op{ATTACHMENT_LOAD_OP_LOAD};
  AttachmentStoreOp stencil_store_op{ATTACHMENT_STORE_OP_STORE};
  ClearDepthStencilValue clear_value{1.0f, 0};
};

struct RenderingInfo {
  std::vector<ColorAttachmentInfo> color_attachments;
  DepthStencilAttachmentInfo depth_stencil;
};

struct Viewport {
  float x{}, y{}, width{}, height{}, min_depth{}, max_depth{1};
};

struct Scissor {
  int32_t x{}, y{};
  uint32_t width{}, height{};
};
class IWindow;

class ICommandContext : public IObject {
 public:
  // AS build/copy/query and DispatchRays require no active rendering scope.
  // Modifying contexts for one AS must be submitted before recording later uses
  // in another context; Build/Update retain inputs and rent scratch automatically.
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
  virtual bool CmdBindRayTracingProgram(IRayTracingProgram *program) = 0;
  // nullptr selects the program default; custom tables must belong to the bound program.
  virtual bool CmdDispatchRays(uint32_t width,
                               uint32_t height,
                               uint32_t depth = 1,
                               IShaderBindingTable *sbt = nullptr) = 0;
  // Cmd calls encode native commands immediately and retain the exact resources
  // they use. Parameter values are consumed during the call, not during Submit.
  // Callers must not use an image concurrently from multiple command contexts.
  // Submit restores every used resource to its backend default state and ends
  // recording; a context cannot be recorded or submitted again.
  // Upload snapshots the CPU bytes now; staging storage is leased through GPU completion.
  // Bind a Program before vertex buffers; rebind buffers after changing its layout.
  // Begin/End delimit an explicit, non-nested rendering scope. Draw requires a scope.
  // Upload, copy, standalone clear, Present and Submit require no active scope.
  // Begin consumes attachment operations; backend attachment leases remain automatic.
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
  // One active graphics or compute program. Switching programs clears group bindings.
  // Dispatch requires no rendering scope; dependencies between dispatches are automatic.
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
  // Clears color, or depth and stencil (if present), according to the image format.
  virtual bool CmdClearImage(IImage *image, const ClearValue &value) = 0;
  // Presents the target window's next image fixed at this call. Multiple windows
  // may be presented by one context; a preceding clear is not required.
  virtual bool CmdPresent(IWindow *window) = 0;

 protected:
  ~ICommandContext() override;
};

}  // namespace grassland::rhi
