#include "grassland/rhi/backend/backend_command_context.h"

namespace grassland::rhi::backend {
ICommandContext::ICommandContext(const void *identity, QueueType type) : identity_(identity), type_(type) {
}

ICommandContext::~ICommandContext() {
  if (image_ && !submitted_)
    image_->CancelClaim();
}

bool ICommandContext::Clear(const ObjectPtr<IImage> &image, const std::array<float, 4> &color) {
  if (sealed_ || failed_ || image_ || type_ != QueueType::Graphics || !image || !image->IsValid() ||
      image->DeviceIdentity() != identity_ || !image->Claim()) {
    ReportError(-1, "ClearColor requires a fresh frame from this device and a graphics context.");
    failed_ = true;
    return false;
  }
  image_ = image;
  color_ = color;
  return true;
}

bool ICommandContext::Present(const ObjectPtr<IImage> &image) {
  if (sealed_ || failed_ || present_ || !image_ || image_.Get() != image.Get() || !image->IsValid()) {
    ReportError(-1, "Present must follow ClearColor for the same acquired frame.");
    failed_ = true;
    return false;
  }
  present_ = true;
  return true;
}

bool ICommandContext::Seal() {
  if (sealed_ || failed_ || (image_ && !present_)) {
    ReportError(-1, "Cannot finish an invalid or incomplete presentation context.");
    return false;
  }
  sealed_ = true;
  return true;
}

const void *ICommandContext::DeviceIdentity() const {
  return identity_;
}

QueueType ICommandContext::Type() const {
  return type_;
}

const ObjectPtr<IImage> &ICommandContext::Frame() const {
  return image_;
}

const std::array<float, 4> &ICommandContext::Color() const {
  return color_;
}

void ICommandContext::MarkSubmitted() {
  submitted_ = true;
}

bool ICommandContext::WasSubmitted() const {
  return submitted_;
}
}  // namespace grassland::rhi::backend
