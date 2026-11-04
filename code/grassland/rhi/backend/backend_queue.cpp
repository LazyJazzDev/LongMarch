#include "grassland/rhi/backend/backend_queue.h"

#include <system_error>

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
  return queue && Owner() == queue->Owner() && Owner()->SharesNativeQueue(type_, queue->type_);
}

ICommandContext *Queue::CreateCommandContext() {
  return Owner()->CreateContextForQueue(type_);
}

rhi::ISubmissionToken *Queue::Submit(rhi::ICommandContext *commands, const rhi::SubmitInfo &info) {
#if !defined(NDEBUG)
  const auto impl = dynamic_cast<CommandContext *>(commands);
  if (!impl || impl->DeviceIdentity() != Owner()->NativeIdentity()) {
    ReportError(-1, "Commands must be submitted to a queue from their device.");
    return nullptr;
  }
#else
  const auto impl = static_cast<CommandContext *>(commands);
#endif
  try {
    std::vector<ObjectPtr<Event>> waits;
    waits.reserve(info.waits.size());
    for (auto token : info.waits) {
#if !defined(NDEBUG)
      auto state = dynamic_cast<Event *>(token);
#else
      auto state = static_cast<Event *>(token);
#endif
      waits.push_back(ObjectPtr<Event>::Retain(state));
    }
    return Owner()->Submit(ObjectPtr<CommandContext>::Retain(impl), waits, type_).Detach();
  } catch (const std::system_error &error) {
    ReportError(error.code().value(), error.what());
  }
  return nullptr;
}

}  // namespace grassland::rhi::backend
