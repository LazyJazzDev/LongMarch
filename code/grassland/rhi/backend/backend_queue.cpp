#include "grassland/rhi/backend/backend_queue.h"

namespace grassland::rhi::backend {

Queue::Queue(Device *device, QueueType type) : device_(device), type_(type) {
}

Queue::~Queue() = default;

Device *Queue::Owner() const {
  return device_;
}

QueueType Queue::GetType() const {
  return type_;
}

bool Queue::SharesPhysicalQueue(const rhi::IQueue *other) const {
  const auto queue = dynamic_cast<const Queue *>(other);
  return queue && Owner() == queue->Owner();
}

}  // namespace grassland::rhi::backend
