#include "grassland/rhi/backend/backend_queue.h"

namespace grassland::rhi::backend {

Queue::Queue(Device *device, QueueType type) : device_(device), type_(type) {
}

Queue::~Queue() = default;

std::mutex &Queue::QueueMutex() {
  return mutex_;
}

uint64_t Queue::NextSerial() {
  return ++serial_;
}

Device *Queue::Owner() const {
  return device_;
}

QueueType Queue::GetType() const {
  return type_;
}

bool Queue::SharesPhysicalQueue(const rhi::IQueue *other) const {
  return this == other;
}

}  // namespace grassland::rhi::backend
