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
  return *this ? Get()->GetStatus() : rhi::SubmissionStatus::Canceled;
}

bool SubmissionToken::WaitHost() const {
  return *this && Get()->WaitHost();
}

rhi::QueueType Queue::GetType() const {
  return *this ? Get()->GetType() : rhi::QueueType::Graphics;
}

bool Queue::SharesPhysicalQueue(const Queue &other) const {
  return *this && Get()->SharesPhysicalQueue(other.Get());
}

SubmissionToken Queue::Submit(RecordedCommands &&commands, const SubmitInfo &info) const {
  RecordedCommands consumed = std::move(commands);
  if (!*this || !consumed)
    return {};
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
