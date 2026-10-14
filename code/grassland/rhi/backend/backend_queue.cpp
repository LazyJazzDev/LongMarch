#include "grassland/rhi/backend/backend_queue.h"

namespace grassland::rhi::backend {

IQueue::IQueue(IDevice *device, QueueType type) : device_(device), type_(type) {
}

IQueue::~IQueue() = default;

IDevice *IQueue::Owner() const {
  return device_;
}

QueueType IQueue::Type() const {
  return type_;
}

}  // namespace grassland::rhi::backend
