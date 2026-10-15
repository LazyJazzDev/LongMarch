#include "grassland/rhi/backend/backend_command_context.h"

namespace grassland::rhi::backend {
CommandContext::CommandContext(const void *identity, QueueType type) : identity_(identity), type_(type) {
}

CommandContext::~CommandContext() {
  if (image_ && !submitted_)
    image_->CancelClaim();
}

bool CommandContext::ClearColor(rhi::IImage *image, const std::array<float, 4> &color) {
  return Clear(ObjectPtr<Image>::Retain(dynamic_cast<Image *>(image)), color);
}

bool CommandContext::Present(rhi::IImage *image) {
  return Present(ObjectPtr<Image>::Retain(dynamic_cast<Image *>(image)));
}

rhi::IRecordedCommands *CommandContext::Finish() {
  if (!Seal())
    return nullptr;
  AddRef();
  return this;
}

bool CommandContext::CanSubmit() const {
  return sealed_ && !failed_ && !submitted_;
}

bool CommandContext::Clear(const ObjectPtr<Image> &image, const std::array<float, 4> &color) {
  if (sealed_)
    return false;
  if (failed_ || image_ || type_ != QUEUE_TYPE_GRAPHICS || !image || !image->IsValid() ||
      image->DeviceIdentity() != identity_ || !image->Claim()) {
    ReportError(-1, "ClearColor requires a fresh frame from this device and a graphics context.");
    failed_ = true;
    return false;
  }
  image_ = image;
  color_ = color;
  return true;
}

bool CommandContext::Present(const ObjectPtr<Image> &image) {
  if (sealed_)
    return false;
  if (failed_ || present_ || !image_ || image_.Get() != image.Get() || !image->IsValid()) {
    ReportError(-1, "Present must follow ClearColor for the same acquired frame.");
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
