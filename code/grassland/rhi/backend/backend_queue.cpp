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

rhi::ISubmissionToken *Queue::Submit(rhi::IRecordedCommands *commands, const rhi::SubmitInfo &info) {
  const auto impl = dynamic_cast<CommandContext *>(commands);
  if (!impl || impl->Type() != Type() || impl->DeviceIdentity() != Owner()->NativeIdentity()) {
    ReportError(-1, "Commands must be submitted to their recording queue.");
    return nullptr;
  }
  try {
    std::vector<ObjectPtr<Device::SubmissionState>> waits;
    waits.reserve(info.waits.size());
    for (auto token : info.waits)
      waits.push_back(ObjectPtr<Device::SubmissionState>::Retain(dynamic_cast<Device::SubmissionState *>(token)));
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
