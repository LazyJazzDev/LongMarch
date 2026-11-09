#include "grassland/rhi/backend/backend_command_context.h"

#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/rhi_program.h"

namespace grassland::rhi::backend {
bool CommandContext::RecordWindowImGui(IWindow *window, IImage *target) {
#if !defined(NDEBUG)
  auto native = dynamic_cast<Window *>(window);
  auto image = dynamic_cast<Image *>(target);
  if (!native || !image || native->DeviceIdentity() != identity_ || image->DeviceIdentity() != identity_ ||
      !glfw::IsWindowThread() || sealed_ || !rendering_.color_attachments.empty() || rendering_.depth_stencil.image) {
    recording_failed_ = true;
    ReportError(-1,
                "CmdDrawImGui requires a window and target from this device outside rendering on the window thread.");
    return false;
  }
#else
  auto native = static_cast<Window *>(window);
#endif
  const auto format = target->GetDesc().format;
  if (!IsColorFormat(format) || IsSignedIntegerFormat(format) || IsUnsignedIntegerFormat(format)) {
    recording_failed_ = true;
    ReportError(-1, "CmdDrawImGui requires a non-integer color target.");
    return false;
  }
  if (!RetainResource(target))
    return false;
  if (!native->RecordImGui(this, target, COLOR_SPACE_BT709_LINEAR, 1.0f)) {
    recording_failed_ = true;
    return false;
  }
  return true;
}

bool CommandContext::RecordWindowPresentation(IWindow *window, IImage *frame_img, ColorSpace frame_color_space) {
#if !defined(NDEBUG)
  auto native = dynamic_cast<Window *>(window);
  auto image = dynamic_cast<Image *>(frame_img);
  if (!native || !image || native->DeviceIdentity() != identity_ || image->DeviceIdentity() != identity_ ||
      !glfw::IsWindowThread() || sealed_ || !rendering_.color_attachments.empty() || rendering_.depth_stencil.image) {
    recording_failed_ = true;
    ReportError(-1,
                "CmdPresentImage requires a window and image from this device outside rendering on the window thread.");
    return false;
  }
#else
  auto native = static_cast<Window *>(window);
#endif
  if (!RetainResource(frame_img))
    return false;
  if (!native->RecordPresentImage(this, frame_img, frame_color_space)) {
    recording_failed_ = true;
    return false;
  }
  return CmdPresent(window);
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
       ds.stencil_store_op > ATTACHMENT_STORE_OP_DISCARD ||
       !(ds.clear_value.depth >= 0 && ds.clear_value.depth <= 1))) {
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

CommandContext::~CommandContext() = default;

void CommandContext::BindDownloads(SubmissionToken *submission) {
  for (const auto &token : downloads_)
    token->BindSubmission(submission);
}

void CommandContext::ReleaseDownloads() {
  downloads_.clear();
}

IDownloadToken *CommandContext::RegisterDownload(ObjectPtr<DownloadToken> token) {
  downloads_.push_back(token);
  return token.Detach();
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

}  // namespace grassland::rhi::backend
