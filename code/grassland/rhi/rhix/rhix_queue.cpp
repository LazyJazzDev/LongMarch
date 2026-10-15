#include "grassland/rhi/rhix/rhix_queue.h"

#include "grassland/rhi/rhix/rhix_command_context.h"

namespace grassland::rhix {
Queue::Queue() noexcept = default;

Queue::Queue(rhi::IQueue *object) noexcept : ObjectContainer(object) {
}

rhi::IQueue *Queue::Get() const noexcept {
  return static_cast<rhi::IQueue *>(GetObject());
}

SubmissionToken::SubmissionToken() noexcept = default;

SubmissionToken::SubmissionToken(rhi::ISubmissionToken *object) noexcept : ObjectContainer(object) {
}

rhi::ISubmissionToken *SubmissionToken::Get() const noexcept {
  return static_cast<rhi::ISubmissionToken *>(GetObject());
}

rhi::SubmissionStatus SubmissionToken::GetStatus() const {
#if !defined(NDEBUG)
  if (!*this)
    return rhi::SUBMISSION_STATUS_CANCELED;
#endif
  return Get()->GetStatus();
}

bool SubmissionToken::WaitHost() const {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->WaitHost();
}

rhi::QueueType Queue::GetType() const {
#if !defined(NDEBUG)
  if (!*this)
    return rhi::QUEUE_TYPE_GRAPHICS;
#endif
  return Get()->GetType();
}

bool Queue::SharesPhysicalQueue(const Queue &other) const {
#if !defined(NDEBUG)
  if (!*this)
    return false;
#endif
  return Get()->SharesPhysicalQueue(other.Get());
}

SubmissionToken Queue::Submit(CommandContext &&commands, const SubmitInfo &info) const {
  CommandContext consumed = std::move(commands);
#if !defined(NDEBUG)
  if (!*this || !consumed)
    return {};
#endif
  try {
    rhi::SubmitInfo native_info;
    native_info.waits.reserve(info.waits.size());
    for (const auto &token : info.waits)
      native_info.waits.push_back(token.Get());
    return Get()->Submit(consumed.Get(), native_info);
  } catch (const std::bad_alloc &) {
    return {};
  }
}
}  // namespace grassland::rhix
