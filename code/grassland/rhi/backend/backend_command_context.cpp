#include "grassland/rhi/backend/backend_command_context.h"

#include <cstdlib>

#include "grassland/rhi/backend/backend_window.h"

namespace grassland::rhi::backend {
CommandContext::CommandContext(const void *identity, QueueType type) : identity_(identity), type_(type) {
}

CommandContext::~CommandContext() {
  if (image_ && !submitted_)
    image_->CancelClaim();
}

bool CommandContext::CmdClearColor(rhi::IImage *image, const std::array<float, 4> &color) {
  return CmdClearColor(ObjectPtr<Image>::Retain(dynamic_cast<Image *>(image)), color);
}

bool CommandContext::CmdClearColor(const ObjectPtr<Image> &image, const std::array<float, 4> &color) {
  if (sealed_)
    return false;
  if (failed_ || present_ || type_ != QUEUE_TYPE_GRAPHICS || !image || !image->IsValid() ||
      image->DeviceIdentity() != identity_ || (image_ && image_.Get() != image.Get())) {
    ReportError(-1, "CmdClearColor requires this context's frame and a graphics queue.");
    failed_ = true;
    return false;
  }
  const bool first_use = !image_;
  if (first_use) {
    if (!image->Claim()) {
      failed_ = true;
      return false;
    }
    // Retain the exact physical resource before encoding any native use.
    image_ = image;
  }
  if (!RecordClear(image_.Get(), color, first_use)) {
    failed_ = true;
    return false;
  }
  return true;
}

bool CommandContext::CmdPresent(rhi::IWindow *window) {
  if (sealed_)
    return false;
  const auto target = dynamic_cast<Window *>(window);
  if (failed_ || present_ || !image_ || !target || !target->OwnsFrame(image_.Get()) || !image_->IsValid()) {
    ReportError(-1, "CmdPresent requires the window owning the recorded clear frame.");
    failed_ = true;
    return false;
  }
  if (!RecordPresent(image_.Get())) {
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
  if (sealed_ || failed_ || (image_ && !present_)) {
    ReportError(-1, "Cannot finish an invalid or incomplete presentation context.");
    return false;
  }
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

QueueType CommandContext::Type() const {
  return type_;
}

const ObjectPtr<Image> &CommandContext::Frame() const {
  return image_;
}

void CommandContext::MarkSubmitted() {
  submitted_ = true;
}

bool CommandContext::WasSubmitted() const {
  return submitted_;
}
}  // namespace grassland::rhi::backend
