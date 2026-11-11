#include "grassland/rhi/rhix/rhix_queue.h"

#include "grassland/rhi/rhix/rhix_command_context.h"

namespace grassland::rhix {
Queue::Queue() noexcept = default;

Queue::Queue(rhi::IQueue *object) noexcept : ObjectContainer(object) {
}

rhi::IQueue *Queue::Get() const noexcept {
  return rhi::QueryInterfacePtr<rhi::IQueue>(GetObject()).Get();
}

rhi::QueueType Queue::GetType() const {
#if !defined(NDEBUG)
  if (!*this)
    return rhi::QUEUE_TYPE_MAIN;
#endif
  return Get()->GetType();
}

CommandContext Queue::CreateCommandContext() const {
#if !defined(NDEBUG)
  if (!*this)
    return nullptr;
#endif
  return Get()->CreateCommandContext();
}

}  // namespace grassland::rhix
