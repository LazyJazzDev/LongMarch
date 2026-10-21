#pragma once
#include <array>

#include "grassland/rhi/rhi.h"
#include "grassland/rhi/rhix/rhix_acceleration_structure.h"
#include "grassland/rhi/rhix/rhix_binding_group.h"
#include "grassland/rhi/rhix/rhix_buffer.h"
#include "grassland/rhi/rhix/rhix_compute_program.h"
#include "grassland/rhi/rhix/rhix_image.h"
#include "grassland/rhi/rhix/rhix_object_container.h"
#include "grassland/rhi/rhix/rhix_program.h"
#include "grassland/rhi/rhix/rhix_ray_tracing_program.h"

namespace grassland::rhix {
class Window;

struct ColorAttachmentInfo {
  Image image;
  rhi::AttachmentLoadOp load_op{rhi::ATTACHMENT_LOAD_OP_LOAD};
  rhi::AttachmentStoreOp store_op{rhi::ATTACHMENT_STORE_OP_STORE};
  std::array<float, 4> clear_color{};
};

struct RenderingInfo {
  std::vector<ColorAttachmentInfo> colors;
  Image depth_stencil;
  rhi::DepthStencilAttachmentInfo depth_stencil_ops;
};

class CommandContext : public ObjectContainer {
 public:
  CommandContext(rhi::ICommandContext *object) noexcept;
  rhi::ICommandContext *Get() const noexcept;
  CommandContext();
  CommandContext(CommandContext &&) noexcept;
  CommandContext &operator=(CommandContext &&) noexcept;
  CommandContext(const CommandContext &) = delete;
  CommandContext &operator=(const CommandContext &) = delete;
  bool CmdUploadBuffer(const Buffer &buffer, uint64_t offset, const void *data, uint64_t size);
  bool CmdCopyBuffer(const Buffer &source,
                     uint64_t source_offset,
                     const Buffer &destination,
                     uint64_t destination_offset,
                     uint64_t size);
  bool CmdBindGroup(uint32_t index, const BindingGroup &group);
  bool CmdUploadImage(const Image &image, const void *data, uint64_t row_pitch);
  bool CmdBuildBottomLevelAccelerationStructure(
      const BottomLevelAccelerationStructure &destination,
      const std::vector<AccelerationStructureGeometryBuffers> &geometries,
      rhi::AccelerationStructureBuildMode mode = rhi::ACCELERATION_STRUCTURE_BUILD);
  bool CmdBuildTopLevelAccelerationStructure(
      const TopLevelAccelerationStructure &destination,
      const std::vector<RayTracingInstance> &instances,
      rhi::AccelerationStructureBuildMode mode = rhi::ACCELERATION_STRUCTURE_BUILD);
  bool CmdQueryCompactedSize(const AccelerationStructure &structure);
  bool CmdCompactAccelerationStructure(const AccelerationStructure &source, const AccelerationStructure &destination);
  bool CmdBindRayTracingProgram(const RayTracingProgram &program);
  bool CmdDispatchRays(uint32_t width, uint32_t height, uint32_t depth = 1);
  bool CmdBindComputeProgram(const ComputeProgram &program);
  bool CmdDispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1);
  bool CmdBindProgram(const Program &program);
  bool CmdBeginRendering(const Image &image,
                         rhi::AttachmentLoadOp load_op = rhi::ATTACHMENT_LOAD_OP_LOAD,
                         rhi::AttachmentStoreOp store_op = rhi::ATTACHMENT_STORE_OP_STORE,
                         const std::array<float, 4> &clear_color = {});
  bool CmdBeginRendering(const RenderingInfo &info);
  bool CmdEndRendering();
  bool CmdBindVertexBuffer(uint32_t binding, const Buffer &buffer, uint64_t offset = 0);
  bool CmdBindIndexBuffer(const Buffer &buffer,
                          rhi::IndexFormat format = rhi::INDEX_FORMAT_UINT32,
                          uint64_t offset = 0);
  bool CmdSetViewport(const rhi::Viewport &viewport);
  bool CmdSetScissor(const rhi::Scissor &scissor);
  bool CmdDraw(uint32_t vertex_count,
               uint32_t instance_count = 1,
               uint32_t first_vertex = 0,
               uint32_t first_instance = 0);
  bool CmdDrawIndexed(uint32_t index_count,
                      uint32_t instance_count = 1,
                      uint32_t first_index = 0,
                      int32_t vertex_offset = 0,
                      uint32_t first_instance = 0);
  bool CmdClearColor(const Image &image, const std::array<float, 4> &color);
  bool CmdPresent(const Window &window);
};

}  // namespace grassland::rhix
