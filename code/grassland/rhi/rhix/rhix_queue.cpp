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
    return rhi::QUEUE_TYPE_MAIN;
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

CommandContext Queue::CreateCommandContext() const {
#if !defined(NDEBUG)
  if (!*this)
    return nullptr;
#endif
  return Get()->CreateCommandContext();
}

}  // namespace grassland::rhix
