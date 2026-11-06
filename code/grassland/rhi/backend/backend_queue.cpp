#include "grassland/rhi/backend/backend_queue.h"

#include <system_error>

#include "grassland/rhi/backend/backend_device.h"

namespace grassland::rhi::backend {
Queue::Queue(Device *device, QueueType type, ObjectPtr<backend::Timeline> timeline)
    : device_(device),
      type_(type),
      timeline_(timeline ? std::move(timeline) : ObjectPtr<backend::Timeline>::Adopt(new backend::Timeline(device))) {
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

backend::Timeline *Queue::TimelineState() const {
  return timeline_.Get();
}

bool Queue::SharesPhysicalQueue(const IQueue *other) const {
  const auto queue = dynamic_cast<const Queue *>(other);
  return queue && Owner() == queue->Owner() && NativeIdentity() == queue->NativeIdentity();
}

ISubmissionToken *Queue::Submit(CommandContext *commands, const SubmitInfo &info) {
  if (commands->OwnerQueue() != this)
    return nullptr;
  try {
    return Owner()->Submit(ObjectPtr<CommandContext>::Retain(commands), info.waits, this).Detach();
  } catch (const std::system_error &error) {
    ReportError(error.code().value(), error.what());
  }
  return nullptr;
}
}  // namespace grassland::rhi::backend
