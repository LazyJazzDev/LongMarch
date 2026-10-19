#include "grassland/rhi/rhix/rhix_command_context.h"

#include "grassland/rhi/rhix/rhix_window.h"

namespace grassland::rhix {
CommandContext::CommandContext() = default;
CommandContext::CommandContext(CommandContext &&) noexcept = default;
CommandContext &CommandContext::operator=(CommandContext &&) noexcept = default;

CommandContext::CommandContext(rhi::ICommandContext *object) noexcept : ObjectContainer(object) {
}

rhi::ICommandContext *CommandContext::Get() const noexcept {
  return static_cast<rhi::ICommandContext *>(GetObject());
}

bool CommandContext::CmdClearColor(const Image &image, const std::array<float, 4> &color) {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->CmdClearColor(image.Get(), color);
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
                                       const std::array<float, 4> &clear_color) {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->CmdBeginRendering({image.Get(), load_op, store_op, clear_color});
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
}  // namespace grassland::rhix
