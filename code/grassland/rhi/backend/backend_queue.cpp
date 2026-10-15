#include "grassland/rhi/backend/backend_queue.h"

namespace grassland::rhi::backend {

Queue::Queue(Device *device, QueueType type) : device_(device), type_(type) {
}

Queue::~Queue() = default;

Device *Queue::Owner() const {
  return device_;
}

QueueType Queue::GetType() const {
  return Type();
}

bool Queue::SharesPhysicalQueue(const rhi::IQueue *other) const {
  const auto queue = dynamic_cast<const Queue *>(other);
  return queue && Owner() == queue->Owner();
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
    std::vector<ObjectPtr<Device::SubmissionState>> waits;
    waits.reserve(info.waits.size());
    for (auto token : info.waits) {
#if !defined(NDEBUG)
      auto state = dynamic_cast<Device::SubmissionState *>(token);
#else
      auto state = static_cast<Device::SubmissionState *>(token);
#endif
      waits.push_back(ObjectPtr<Device::SubmissionState>::Retain(state));
    }
    return Owner()->Submit(ObjectPtr<CommandContext>::Retain(impl), waits).Detach();
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Submission allocation failed.");
  } catch (const std::system_error &error) {
    ReportError(error.code().value(), error.what());
  }
  return nullptr;
}

QueueType Queue::Type() const {
  return type_;
}

}  // namespace grassland::rhi::backend
