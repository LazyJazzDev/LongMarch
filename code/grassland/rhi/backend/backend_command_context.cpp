#include "grassland/rhi/backend/backend_command_context.h"

#include <cstdlib>

#include "grassland/rhi/backend/backend_window.h"

namespace grassland::rhi::backend {
CommandContext::CommandContext(const void *identity) : identity_(identity) {
}

CommandContext::~CommandContext() = default;

bool CommandContext::RetainResource(rhi::IObject *resource) {
#if !defined(NDEBUG)
  if (sealed_)
    return false;
#endif
  if (failed_)
    return false;
  if (resources_.find(resource) != resources_.end())
    return true;
  try {
    resources_.emplace(resource, ObjectPtr<rhi::IObject>::Retain(resource));
  } catch (const std::bad_alloc &) {
    failed_ = true;
    ReportError(-1, "Command resource retention ran out of memory.");
    return false;
  }
  return true;
}

bool CommandContext::CmdClearColor(rhi::IImage *image, const std::array<float, 4> &color) {
#if !defined(NDEBUG)
  auto native = dynamic_cast<Image *>(image);
#else
  auto native = static_cast<Image *>(image);
#endif
  return CmdClearColor(ObjectPtr<Image>::Retain(native), color);
}

bool CommandContext::CmdClearColor(const ObjectPtr<Image> &image, const std::array<float, 4> &color) {
  if (failed_)
    return false;
#if !defined(NDEBUG)
  if (sealed_)
    return false;
  if (present_ || !image || !image->IsValid() || image->DeviceIdentity() != identity_ ||
      (frame_ && frame_ != image.Get())) {
    ReportError(-1, "CmdClearColor requires this context's frame.");
    failed_ = true;
    return false;
  }
#endif
  const bool first_use = !frame_;
  if (first_use) {
    if (!RetainResource(image.Get()))
      return false;
    frame_ = image.Get();
  }
  if (!RecordClear(frame_, color, first_use)) {
    failed_ = true;
    return false;
  }
  return true;
}

bool CommandContext::CmdPresent(rhi::IWindow *window) {
  if (failed_)
    return false;
#if !defined(NDEBUG)
  if (sealed_)
    return false;
  const auto target = dynamic_cast<Window *>(window);
  if (present_ || !frame_ || !target || !target->OwnsFrame(frame_) || !frame_->IsValid()) {
    ReportError(-1, "CmdPresent requires the window owning the recorded clear frame.");
    failed_ = true;
    return false;
  }
#else
  const auto target = static_cast<Window *>(window);
#endif
  if (!RecordPresent(frame_)) {
    failed_ = true;
    return false;
  }
  present_window_ = target;
  present_ = true;
  return true;
}

void CommandContext::AdvanceFrame() {
  if (!present_window_)
    return;
  if (!present_window_->UpdateNextImage()) {
    ReportError(-1, "Failed to prepare the next swapchain image after submission.");
    std::abort();
  }
}

bool CommandContext::Seal() {
  if (failed_)
    return false;
#if !defined(NDEBUG)
  if (sealed_ || (frame_ && !present_)) {
    ReportError(-1, "Cannot finish an invalid or incomplete presentation context.");
    return false;
  }
#endif
  sealed_ = true;
  if (!EndRecording()) {
    failed_ = true;
    return false;
  }
  return true;
}

const void *CommandContext::DeviceIdentity() const {
  return identity_;
}

Image *CommandContext::Frame() const {
  return frame_;
}

void CommandContext::MarkSubmitted() {
  submitted_ = true;
}

bool CommandContext::WasSubmitted() const {
  return submitted_;
}
}  // namespace grassland::rhi::backend
