#include "grassland/rhi/backend/backend_command_context.h"

#include <system_error>

#include "grassland/rhi/backend/backend_command_context_state.h"
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/rhi_program.h"

namespace grassland::rhi::backend {
bool CommandContext::CmdDrawImGui(IWindow *window, IImage *target) {
  auto native = rhi::QueryInterfacePtr<Window>(window).Get();
  auto image = rhi::QueryInterfacePtr<Image>(target).Get();
  if (!native || !image || native->DeviceIdentity() != DeviceIdentity() ||
      (image->DeviceIdentity() != DeviceIdentity() || !image->SupportsQueue(OwnerQueue())) || !glfw::IsWindowThread() ||
      StateCommandContext().sealed_ || !StateCommandContext().rendering_.color_attachments.empty() ||
      StateCommandContext().rendering_.depth_stencil.image) {
    StateCommandContext().recording_failed_ = true;
    ReportError(-1,
                "CmdDrawImGui requires a window and target from this device outside rendering on the window thread.");
    return false;
  }
  const auto format = target->GetDesc().format;
  if (!IsColorFormat(format) || IsSignedIntegerFormat(format) || IsUnsignedIntegerFormat(format)) {
    StateCommandContext().recording_failed_ = true;
    ReportError(-1, "CmdDrawImGui requires a non-integer color target.");
    return false;
  }
  if (!RetainResource(target))
    return false;
  if (!native->RecordImGui(this, target, COLOR_SPACE_BT709_LINEAR, 1.0f)) {
    StateCommandContext().recording_failed_ = true;
    return false;
  }
  return true;
}

bool CommandContext::CmdPresentImage(IWindow *window, IImage *frame_img, ColorSpace frame_color_space) {
  auto native = rhi::QueryInterfacePtr<Window>(window).Get();
  auto image = rhi::QueryInterfacePtr<Image>(frame_img).Get();
  if (!native || !image || native->DeviceIdentity() != DeviceIdentity() ||
      (image->DeviceIdentity() != DeviceIdentity() || !image->SupportsQueue(OwnerQueue())) || !glfw::IsWindowThread() ||
      StateCommandContext().sealed_ || !StateCommandContext().rendering_.color_attachments.empty() ||
      StateCommandContext().rendering_.depth_stencil.image) {
    StateCommandContext().recording_failed_ = true;
    ReportError(-1,
                "CmdPresentImage requires a window and image from this device outside rendering on the window thread.");
    return false;
  }
  if (!RetainResource(frame_img))
    return false;
  if (!native->RecordPresentImage(this, frame_img, frame_color_space)) {
    StateCommandContext().recording_failed_ = true;
    return false;
  }
  return CmdPresent(window);
}

bool CommandContext::CmdBuildBottomLevelAccelerationStructure(IBottomLevelAccelerationStructure *,
                                                              const std::vector<AccelerationStructureGeometryBuffers> &,
                                                              AccelerationStructureBuildMode) {
  StateCommandContext().recording_failed_ = true;
  return false;
}

bool CommandContext::CmdBuildTopLevelAccelerationStructure(ITopLevelAccelerationStructure *,
                                                           const std::vector<RayTracingInstance> &,
                                                           AccelerationStructureBuildMode) {
  StateCommandContext().recording_failed_ = true;
  return false;
}

bool CommandContext::CmdBindRayTracingProgram(IRayTracingProgram *) {
  StateCommandContext().recording_failed_ = true;
  return false;
}

bool CommandContext::CmdDispatchRays(uint32_t, uint32_t, uint32_t, IShaderBindingTable *) {
  StateCommandContext().recording_failed_ = true;
  return false;
}

bool CommandContext::CmdBindComputeProgram(IComputeProgram *) {
  StateCommandContext().recording_failed_ = true;
  return false;
}

bool CommandContext::CmdDispatch(uint32_t, uint32_t, uint32_t) {
  StateCommandContext().recording_failed_ = true;
  return false;
}

bool CommandContext::CmdBindGroup(uint32_t, IBindingGroup *) {
  StateCommandContext().recording_failed_ = true;
  return false;
}

bool CommandContext::CmdUploadImage(IImage *, const void *, uint64_t) {
  StateCommandContext().recording_failed_ = true;
  return false;
}

bool CommandContext::PrepareRendering(const RenderingInfo &info) {
  auto next = info;
  if (next.color_attachments.size() > 8 || (next.color_attachments.empty() && !next.depth_stencil.image)) {
    StateCommandContext().recording_failed_ = true;
    return false;
  }
  std::vector<IImage *> images;
  auto validate = [&](IImage *image, bool depth, AttachmentLoadOp load, AttachmentStoreOp store) {
    auto native = rhi::QueryInterfacePtr<Image>(image).Get();
    if (!native || !native->IsValid() ||
        (native->DeviceIdentity() != DeviceIdentity() || !native->SupportsQueue(OwnerQueue())) ||
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
      StateCommandContext().recording_failed_ = true;
      return false;
    }
  const auto &ds = next.depth_stencil;
  if (ds.image &&
      (!validate(ds.image, true, ds.depth_load_op, ds.depth_store_op) || ds.stencil_load_op < ATTACHMENT_LOAD_OP_LOAD ||
       ds.stencil_load_op > ATTACHMENT_LOAD_OP_DISCARD || ds.stencil_store_op < ATTACHMENT_STORE_OP_STORE ||
       ds.stencil_store_op > ATTACHMENT_STORE_OP_DISCARD ||
       !(ds.clear_value.depth >= 0 && ds.clear_value.depth <= 1))) {
    StateCommandContext().recording_failed_ = true;
    return false;
  }
  StateCommandContext().rendering_ = std::move(next);
  return true;
}

bool CommandContext::RenderingMatches(const IProgram *program) const {
  const auto &desc = program->GetDesc();
  if (desc.color_formats.size() != StateCommandContext().rendering_.color_attachments.size())
    return false;
  for (size_t i = 0; i < desc.color_formats.size(); ++i)
    if (desc.color_formats[i] !=
        StateCommandContext().rendering_.color_attachments[i].color_attachment->GetDesc().format)
      return false;
  return desc.depth_stencil.format == (StateCommandContext().rendering_.depth_stencil.image
                                           ? StateCommandContext().rendering_.depth_stencil.image->GetDesc().format
                                           : IMAGE_FORMAT_UNKNOWN);
}

void CommandContext::InitializeCommandContext(Queue *queue) {
  StateCommandContext().queue_ = queue;
}

rhi::ISubmissionToken *CommandContext::Submit(const rhi::SubmitInfo &info) {
  try {
    return StateCommandContext().queue_->Submit(ObjectPtr<CommandContext>::Retain(this), info).Detach();
  } catch (const std::system_error &error) {
    ReportError(error.code().value(), error.what());
  }
  return nullptr;
}

void CommandContext::BindDownloads(SubmissionToken *submission) {
  for (const auto &token : StateCommandContext().downloads_)
    token->BindSubmission(submission);
}

void CommandContext::ReleaseDownloads() {
  StateCommandContext().downloads_.clear();
}

IDownloadToken *CommandContext::RegisterDownload(ObjectPtr<DownloadToken> token) {
  StateCommandContext().downloads_.push_back(token);
  return token.Detach();
}

IDownloadToken *CommandContext::CmdDownloadBuffer(IBuffer *, uint64_t, uint64_t) {
  StateCommandContext().recording_failed_ = true;
  return nullptr;
}

IDownloadToken *CommandContext::CmdDownloadImage(IImage *, const ImageDownloadRegion &) {
  StateCommandContext().recording_failed_ = true;
  return nullptr;
}

bool CommandContext::RetainResource(rhi::IObject *resource) {
#if !defined(NDEBUG)
  if (StateCommandContext().sealed_)
    return false;
#endif
  if (StateCommandContext().recording_failed_)
    return false;
  if (StateCommandContext().resources_.find(resource) != StateCommandContext().resources_.end())
    return true;
  StateCommandContext().resources_.emplace(resource, ObjectPtr<rhi::IObject>::Retain(resource));

  return true;
}

bool CommandContext::Seal() {
  if (StateCommandContext().recording_failed_)
    return false;
#if !defined(NDEBUG)
  if (StateCommandContext().sealed_) {
    ReportError(-1, "Cannot submit a command context twice.");
    return false;
  }
#endif
  StateCommandContext().sealed_ = true;
  if (!EndRecording()) {
    StateCommandContext().recording_failed_ = true;
    return false;
  }
  return true;
}

Queue *CommandContext::OwnerQueue() const {
  return StateCommandContext().queue_;
}

const void *CommandContext::DeviceIdentity() const {
  return StateCommandContext().queue_->Owner();
}

void CommandContext::MarkSubmitted() {
  StateCommandContext().submitted_ = true;
}

bool CommandContext::WasSubmitted() const {
  return StateCommandContext().submitted_;
}

bool CommandContext::CmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size) {
  StateCommandContext().recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdCopyBuffer(IBuffer *source,
                                   uint64_t source_offset,
                                   IBuffer *destination,
                                   uint64_t destination_offset,
                                   uint64_t size) {
  StateCommandContext().recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdBindProgram(IProgram *program) {
  StateCommandContext().recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdBeginRendering(const RenderingInfo &info) {
  StateCommandContext().recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdEndRendering() {
  StateCommandContext().recording_failed_ = true;
  ReportError(-1, "Drawing is not implemented for this backend.");
  return false;
}

bool CommandContext::CmdBindVertexBuffer(uint32_t binding, IBuffer *buffer, uint64_t offset) {
  StateCommandContext().recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdBindIndexBuffer(IBuffer *buffer, IndexFormat format, uint64_t offset) {
  StateCommandContext().recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdSetViewport(const Viewport &viewport) {
  StateCommandContext().recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdSetScissor(const Scissor &scissor) {
  StateCommandContext().recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdDraw(uint32_t vertex_count,
                             uint32_t instance_count,
                             uint32_t first_vertex,
                             uint32_t first_instance) {
  StateCommandContext().recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}

bool CommandContext::CmdDrawIndexed(uint32_t index_count,
                                    uint32_t instance_count,
                                    uint32_t first_index,
                                    int32_t vertex_offset,
                                    uint32_t first_instance) {
  StateCommandContext().recording_failed_ = true;
  ReportError(-1, "Drawing and buffers are not implemented for this backend.");
  return false;
}
}  // namespace grassland::rhi::backend

namespace grassland::rhi::backend {
bool CommandContext::CmdQueryCompactedSize(IAccelerationStructure *) {
  StateCommandContext().recording_failed_ = true;
  return false;
}

bool CommandContext::CmdCompactAccelerationStructure(IAccelerationStructure *, IAccelerationStructure *) {
  StateCommandContext().recording_failed_ = true;
  return false;
}
}  // namespace grassland::rhi::backend

LONGMARCH_RHI_UUID(grassland::rhi::backend::CommandContext, "4a13943e-6f88-4739-91c4-0eb6b6a04a0c")

namespace grassland::rhi::backend {
CommandContextState::CommandContextState() = default;
CommandContextState::~CommandContextState() = default;
}  // namespace grassland::rhi::backend
