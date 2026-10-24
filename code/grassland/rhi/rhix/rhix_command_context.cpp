#include "grassland/rhi/rhix/rhix_command_context.h"

#include "grassland/rhi/rhix/rhix_window.h"

namespace grassland::rhix {
bool CommandContext::CmdBeginRendering(const RenderingInfo &info) {
  rhi::RenderingInfo native;
  for (const auto &color : info.colors)
    native.color_attachments.push_back({color.image.Get(), color.load_op, color.store_op, color.clear_value});
  native.depth_stencil = info.depth_stencil_ops;
  native.depth_stencil.image = info.depth_stencil.Get();
  return Get()->CmdBeginRendering(native);
}

CommandContext::CommandContext() = default;
CommandContext::CommandContext(CommandContext &&) noexcept = default;
CommandContext &CommandContext::operator=(CommandContext &&) noexcept = default;

CommandContext::CommandContext(rhi::ICommandContext *object) noexcept : ObjectContainer(object) {
}

rhi::ICommandContext *CommandContext::Get() const noexcept {
  return static_cast<rhi::ICommandContext *>(GetObject());
}

bool CommandContext::CmdClearImage(const Image &image, const rhi::ClearValue &value) {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->CmdClearImage(image.Get(), value);
}

bool CommandContext::CmdPresent(const Window &window) {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->CmdPresent(window.Get());
}

bool CommandContext::CmdUploadBuffer(const Buffer &buffer, uint64_t offset, const void *data, uint64_t size) {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->CmdUploadBuffer(buffer.Get(), offset, data, size);
}

bool CommandContext::CmdCopyBuffer(const Buffer &source,
                                   uint64_t source_offset,
                                   const Buffer &destination,
                                   uint64_t destination_offset,
                                   uint64_t size) {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->CmdCopyBuffer(source.Get(), source_offset, destination.Get(), destination_offset, size);
}

bool CommandContext::CmdBindComputeProgram(const ComputeProgram &program) {
  return Get()->CmdBindComputeProgram(program.Get());
}

bool CommandContext::CmdDispatch(uint32_t x, uint32_t y, uint32_t z) {
  return Get()->CmdDispatch(x, y, z);
}

bool CommandContext::CmdBindProgram(const Program &program) {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->CmdBindProgram(program.Get());
}

bool CommandContext::CmdBeginRendering(const Image &image,
                                       rhi::AttachmentLoadOp load_op,
                                       rhi::AttachmentStoreOp store_op,
                                       const rhi::ClearValue &clear_value) {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->CmdBeginRendering({{{image.Get(), load_op, store_op, clear_value}}});
}

bool CommandContext::CmdEndRendering() {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->CmdEndRendering();
}

bool CommandContext::CmdBindVertexBuffer(uint32_t binding, const Buffer &buffer, uint64_t offset) {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->CmdBindVertexBuffer(binding, buffer.Get(), offset);
}

bool CommandContext::CmdBindIndexBuffer(const Buffer &buffer, rhi::IndexFormat format, uint64_t offset) {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->CmdBindIndexBuffer(buffer.Get(), format, offset);
}

bool CommandContext::CmdSetViewport(const rhi::Viewport &viewport) {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->CmdSetViewport(viewport);
}

bool CommandContext::CmdSetScissor(const rhi::Scissor &scissor) {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->CmdSetScissor(scissor);
}

bool CommandContext::CmdDraw(uint32_t vertex_count,
                             uint32_t instance_count,
                             uint32_t first_vertex,
                             uint32_t first_instance) {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->CmdDraw(vertex_count, instance_count, first_vertex, first_instance);
}

bool CommandContext::CmdDrawIndexed(uint32_t index_count,
                                    uint32_t instance_count,
                                    uint32_t first_index,
                                    int32_t vertex_offset,
                                    uint32_t first_instance) {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->CmdDrawIndexed(index_count, instance_count, first_index, vertex_offset, first_instance);
}

bool CommandContext::CmdBindGroup(uint32_t index, const BindingGroup &group) {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->CmdBindGroup(index, group.Get());
}

bool CommandContext::CmdUploadImage(const Image &image, const void *data, uint64_t row_pitch) {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->CmdUploadImage(image.Get(), data, row_pitch);
}

bool CommandContext::CmdBuildBottomLevelAccelerationStructure(
    const BottomLevelAccelerationStructure &destination,
    const std::vector<AccelerationStructureGeometryBuffers> &geometries,
    rhi::AccelerationStructureBuildMode mode) {
  std::vector<rhi::AccelerationStructureGeometryBuffers> native;
  for (const auto &g : geometries)
    native.push_back({{g.data.buffer.Get(), g.data.offset, g.data.size},
                      {g.indices.buffer.Get(), g.indices.offset, g.indices.size}});
  return Get()->CmdBuildBottomLevelAccelerationStructure(destination.Get(), native, mode);
}

bool CommandContext::CmdBuildTopLevelAccelerationStructure(const TopLevelAccelerationStructure &destination,
                                                           const std::vector<RayTracingInstance> &instances,
                                                           rhi::AccelerationStructureBuildMode mode) {
  std::vector<rhi::RayTracingInstance> native;
  for (const auto &i : instances)
    native.push_back({i.blas.Get(), i.transform, i.custom_index, i.hit_group_offset, i.mask, i.disable_triangle_culling,
                      i.front_counter_clockwise});
  return Get()->CmdBuildTopLevelAccelerationStructure(destination.Get(), native, mode);
}

bool CommandContext::CmdQueryCompactedSize(const AccelerationStructure &structure) {
  return Get()->CmdQueryCompactedSize(structure.Get());
}

bool CommandContext::CmdCompactAccelerationStructure(const AccelerationStructure &source,
                                                     const AccelerationStructure &destination) {
  return Get()->CmdCompactAccelerationStructure(source.Get(), destination.Get());
}

bool CommandContext::CmdBindRayTracingProgram(const RayTracingProgram &program) {
  return Get()->CmdBindRayTracingProgram(program.Get());
}

bool CommandContext::CmdDispatchRays(uint32_t width, uint32_t height, uint32_t depth, const ShaderBindingTable &sbt) {
  return Get()->CmdDispatchRays(width, height, depth, sbt.Get());
}

DownloadToken CommandContext::CmdDownloadBuffer(const Buffer &buffer, uint64_t offset, uint64_t size) {
  return *this ? Get()->CmdDownloadBuffer(buffer.Get(), offset, size) : nullptr;
}

DownloadToken CommandContext::CmdDownloadImage(const Image &image, const rhi::ImageDownloadRegion &region) {
  return *this ? Get()->CmdDownloadImage(image.Get(), region) : nullptr;
}

}  // namespace grassland::rhix
