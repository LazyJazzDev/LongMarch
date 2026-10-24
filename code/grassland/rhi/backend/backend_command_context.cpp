#include "grassland/rhi/backend/backend_command_context.h"

#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/rhi_program.h"

namespace grassland::rhi::backend {
bool CommandContext::CmdBuildBottomLevelAccelerationStructure(IBottomLevelAccelerationStructure *,
                                                              const std::vector<AccelerationStructureGeometryBuffers> &,
                                                              AccelerationStructureBuildMode) {
  recording_failed_ = true;
  return false;
}

bool CommandContext::CmdBuildTopLevelAccelerationStructure(ITopLevelAccelerationStructure *,
                                                           const std::vector<RayTracingInstance> &,
                                                           AccelerationStructureBuildMode) {
  recording_failed_ = true;
  return false;
}

bool CommandContext::CmdBindRayTracingProgram(IRayTracingProgram *) {
  recording_failed_ = true;
  return false;
}

bool CommandContext::CmdDispatchRays(uint32_t, uint32_t, uint32_t, IShaderBindingTable *) {
  recording_failed_ = true;
  return false;
}

bool CommandContext::CmdBindComputeProgram(IComputeProgram *) {
  recording_failed_ = true;
  return false;
}

bool CommandContext::CmdDispatch(uint32_t, uint32_t, uint32_t) {
  recording_failed_ = true;
  return false;
}

bool CommandContext::CmdBindGroup(uint32_t, IBindingGroup *) {
  recording_failed_ = true;
  return false;
}

bool CommandContext::CmdUploadImage(IImage *, const void *, uint64_t) {
  recording_failed_ = true;
  return false;
}

bool CommandContext::PrepareRendering(const RenderingInfo &info) {
  auto next = info;
  if (next.color_attachments.size() > 8 || (next.color_attachments.empty() && !next.depth_stencil.image)) {
    recording_failed_ = true;
    return false;
  }
  std::vector<IImage *> images;
  auto validate = [&](IImage *image, bool depth, AttachmentLoadOp load, AttachmentStoreOp store) {
    auto native = dynamic_cast<Image *>(image);
    if (!native || !native->IsValid() || native->DeviceIdentity() != DeviceIdentity() ||
        IsDepthFormat(image->GetDesc().format) != depth || load < ATTACHMENT_LOAD_OP_LOAD ||
        load > ATTACHMENT_LOAD_OP_DISCARD || store < ATTACHMENT_STORE_OP_STORE || store > ATTACHMENT_STORE_OP_DISCARD)
      return false;
    for (auto previous : images)
      if (previous == image || previous->GetDesc().width != image->GetDesc().width ||
          previous->GetDesc().height != image->GetDesc().height)
        return false;
    images.push_back(image);
    return RetainResource(image);
  };
  for (auto color : next.color_attachments)
    if (!validate(color.color_attachment, false, color.load_op, color.store_op)) {
      recording_failed_ = true;
      return false;
    }
  const auto &ds = next.depth_stencil;
  if (ds.image &&
      (!validate(ds.image, true, ds.depth_load_op, ds.depth_store_op) || ds.stencil_load_op < ATTACHMENT_LOAD_OP_LOAD ||
       ds.stencil_load_op > ATTACHMENT_LOAD_OP_DISCARD || ds.stencil_store_op < ATTACHMENT_STORE_OP_STORE ||
       ds.stencil_store_op > ATTACHMENT_STORE_OP_DISCARD || !(ds.clear_depth >= 0 && ds.clear_depth <= 1))) {
    recording_failed_ = true;
    return false;
  }
  rendering_ = std::move(next);
  return true;
}

bool CommandContext::RenderingMatches(const IProgram *program) const {
  const auto &desc = program->GetDesc();
  if (desc.color_formats.size() != rendering_.color_attachments.size())
    return false;
  for (size_t i = 0; i < desc.color_formats.size(); ++i)
    if (desc.color_formats[i] != rendering_.color_attachments[i].color_attachment->GetDesc().format)
      return false;
  return desc.depth_stencil.format ==
         (rendering_.depth_stencil.image ? rendering_.depth_stencil.image->GetDesc().format : IMAGE_FORMAT_UNKNOWN);
}

CommandContext::CommandContext(const void *identity) : identity_(identity) {
}

CommandContext::~CommandContext() {
  CompleteDownloads(SUBMISSION_STATUS_CANCELED);
}

void CommandContext::CompleteDownloads(SubmissionStatus status) {
  for (auto &token : downloads_)
    token->Complete(status);
  downloads_.clear();
}

IDownloadToken *CommandContext::TrackDownload(uint64_t size, std::function<bool(void *)> copy) {
  auto token = ObjectPtr<DownloadToken>::Adopt(new DownloadToken(size, std::move(copy)));
  downloads_.push_back(token);
  return token.Detach();
}

IDownloadToken *CommandContext::CmdDownloadBuffer(IBuffer *, uint64_t, uint64_t) {
  recording_failed_ = true;
  return nullptr;
}

IDownloadToken *CommandContext::CmdDownloadImage(IImage *, const ImageDownloadRegion &) {
  recording_failed_ = true;
  return nullptr;
}

bool CommandContext::RetainResource(rhi::IObject *resource) {
#if !defined(NDEBUG)
  if (sealed_)
    return false;
#endif
  if (recording_failed_)
    return false;
  if (resources_.find(resource) != resources_.end())
    return true;
  resources_.emplace(resource, ObjectPtr<rhi::IObject>::Retain(resource));

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

bool CommandContext::CmdBeginRendering(const RenderingInfo &info) {
  recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdEndRendering() {
  recording_failed_ = true;
  ReportError(-1, "Drawing is not implemented for this backend.");
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

namespace grassland::rhi::backend {
bool CommandContext::CmdQueryCompactedSize(IAccelerationStructure *) {
  recording_failed_ = true;
  return false;
}

bool CommandContext::CmdCompactAccelerationStructure(IAccelerationStructure *, IAccelerationStructure *) {
  recording_failed_ = true;
  return false;
}
}  // namespace grassland::rhi::backend
