#include "grassland/rhi/backend/backend_command_context.h"

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
  if (failed_ || image_ || type_ != QUEUE_TYPE_GRAPHICS || !image || !image->IsValid() ||
      image->DeviceIdentity() != identity_ || !image->Claim()) {
    ReportError(-1, "CmdClearColor requires a fresh frame from this device and a graphics context.");
    failed_ = true;
    return false;
  }
  image_ = image;
  color_ = color;
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
  present_ = true;
  return true;
}

bool CommandContext::Seal() {
  if (sealed_ || failed_ || (image_ && !present_)) {
    ReportError(-1, "Cannot finish an invalid or incomplete presentation context.");
    return false;
  }
  sealed_ = true;
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

const std::array<float, 4> &CommandContext::Color() const {
  return color_;
}

void CommandContext::MarkSubmitted() {
  submitted_ = true;
}

bool CommandContext::WasSubmitted() const {
  return submitted_;
}
}  // namespace grassland::rhi::backend
