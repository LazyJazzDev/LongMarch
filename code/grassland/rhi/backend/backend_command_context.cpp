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
}  // namespace grassland::rhi::backend
