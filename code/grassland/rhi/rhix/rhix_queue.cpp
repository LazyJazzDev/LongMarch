#include "grassland/rhi/rhix/rhix_queue.h"

#include "grassland/rhi/rhix/rhix_command_context.h"

namespace grassland::rhix {
Queue::Queue() noexcept = default;

Queue::Queue(rhi::IQueue *object) noexcept : ObjectContainer(object) {
}

rhi::IQueue *Queue::Get() const noexcept {
  return static_cast<rhi::IQueue *>(GetObject());
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
  rhi::SubmitInfo native_info;
  native_info.waits.reserve(info.waits.size());
  for (const auto &token : info.waits)
    native_info.waits.push_back(token.Get());
#if defined(LONGMARCH_CUDA_RUNTIME)
  for (const auto &point : info.cuda_waits)
    native_info.cuda_waits.push_back({point.timeline.Get(), point.value});
  for (const auto &point : info.cuda_signals)
    native_info.cuda_signals.push_back({point.timeline.Get(), point.value});
#endif
  return Get()->Submit(consumed.Get(), native_info);
}
}  // namespace grassland::rhix
