#include "grassland/rhi/backend/backend_command_context.h"

namespace grassland::rhi::backend {
CommandContext::CommandContext(const void *identity) : identity_(identity) {
}

CommandContext::~CommandContext() = default;

bool CommandContext::RetainResource(rhi::IObject *resource) {
#if !defined(NDEBUG)
  if (sealed_)
    return false;
#endif
  if (recording_failed_)
    return false;
  if (resources_.find(resource) != resources_.end())
    return true;
  try {
    resources_.emplace(resource, ObjectPtr<rhi::IObject>::Retain(resource));
  } catch (const std::bad_alloc &) {
    recording_failed_ = true;
    ReportError(-1, "Command resource retention ran out of memory.");
    return false;
  }
  return true;
}

bool CommandContext::Seal() {
  if (recording_failed_)
    return false;
#if !defined(NDEBUG)
  if (sealed_) {
    ReportError(-1, "Cannot submit a command context twice.");
    return false;
  }
#endif
  sealed_ = true;
  if (!EndRecording()) {
    recording_failed_ = true;
    return false;
  }
  return true;
}

const void *CommandContext::DeviceIdentity() const {
  return identity_;
}

void CommandContext::MarkSubmitted() {
  submitted_ = true;
}

bool CommandContext::WasSubmitted() const {
  return submitted_;
}

bool CommandContext::CmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size) {
  recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdCopyBuffer(IBuffer *source,
                                   uint64_t source_offset,
                                   IBuffer *destination,
                                   uint64_t destination_offset,
                                   uint64_t size) {
  recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdBindProgram(IProgram *program) {
  recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdBindRenderTarget(IImage *image) {
  recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdBindVertexBuffer(uint32_t binding, IBuffer *buffer, uint64_t offset) {
  recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdBindIndexBuffer(IBuffer *buffer, IndexFormat format, uint64_t offset) {
  recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdSetViewport(const Viewport &viewport) {
  recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdSetScissor(const Scissor &scissor) {
  recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdDraw(uint32_t vertex_count,
                             uint32_t instance_count,
                             uint32_t first_vertex,
                             uint32_t first_instance) {
  recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdDrawIndexed(uint32_t index_count,
                                    uint32_t instance_count,
                                    uint32_t first_index,
                                    int32_t vertex_offset,
                                    uint32_t first_instance) {
  recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}
}  // namespace grassland::rhi::backend
