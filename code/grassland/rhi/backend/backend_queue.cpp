#include "grassland/rhi/backend/backend_queue.h"

#include <system_error>

#include "grassland/rhi/backend/backend_device.h"

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

const void *Queue::NativeIdentity() const {
  return device_;
}

std::mutex &Queue::Mutex() {
  return mutex_;
}

uint64_t Queue::NextSerial() {
  return ++serial_;
}

bool Queue::HasNativeTimeline() const {
  return false;
}

bool Queue::QueryTimeline(uint64_t &value) const {
  value = completed_.load();
  return true;
}

void Queue::RecordCompletion(uint64_t value) {
  completed_.store(value);
}

bool Queue::SharesPhysicalQueue(const IQueue *other) const {
  const auto queue = dynamic_cast<const Queue *>(other);
  return queue && Owner() == queue->Owner() && NativeIdentity() == queue->NativeIdentity();
}

ISubmissionToken *Queue::Submit(CommandContext *commands, const SubmitInfo &info) {
  if (commands->OwnerQueue() != this)
    return nullptr;
  try {
    std::vector<ObjectPtr<Event>> waits;
    for (auto event : info.waits) {
      auto native = dynamic_cast<Event *>(event);
      if (!native || native->Owner() != Owner())
        return nullptr;
      waits.push_back(ObjectPtr<Event>::Retain(native));
    }
    return Owner()->Submit(ObjectPtr<CommandContext>::Retain(commands), waits, this).Detach();
  } catch (const std::system_error &error) {
    ReportError(error.code().value(), error.what());
  }
  return nullptr;
}
}  // namespace grassland::rhi::backend
