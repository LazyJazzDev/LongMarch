#include "grassland/rhi/backend/backend_device.h"
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/rhi_device.h"

namespace grassland::rhi {

Queue::Queue() noexcept = default;
SubmissionToken::SubmissionToken() noexcept = default;

using namespace backend;

SubmissionStatus SubmissionToken::GetStatus() const {
  return *this ? Access::As<IDevice::SubmissionState>(*this).Status() : SubmissionStatus::Canceled;
}

bool SubmissionToken::WaitHost() const {
  return *this && Access::As<IDevice::SubmissionState>(*this).Wait();
}

SubmissionToken Queue::Submit(RecordedCommands &&commands, const SubmitInfo &info) const {
  auto consumed = Access::Retain<ICommandContext>(commands);
  commands = {};
  if (!*this || !consumed)
    return {};
  const auto &queue = Access::As<IQueue>(*this);
  if (queue.Type() != consumed->Type() || queue.Owner()->NativeIdentity() != consumed->DeviceIdentity()) {
    ReportError(-1, "Commands must be submitted to their recording queue.");
    return {};
  }
  try {
    std::vector<ObjectPtr<IDevice::SubmissionState>> waits;
    waits.reserve(info.waits.size());
    for (const auto &token : info.waits)
      waits.push_back(Access::Retain<IDevice::SubmissionState>(token));
    return Access::Share<SubmissionToken>(queue.Owner()->Submit(consumed, waits));
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Submission allocation failed.");
  } catch (const std::system_error &error) {
    ReportError(error.code().value(), error.what());
  }
  return {};
}

QueueType Queue::GetType() const {
  return *this ? Access::As<IQueue>(*this).Type() : QueueType::Graphics;
}

bool Queue::SharesPhysicalQueue(const Queue &other) const {
  // This first implementation deliberately maps both roles to one graphics queue.
  return *this && other && Access::As<IQueue>(*this).Owner() == Access::As<IQueue>(other).Owner();
}

}  // namespace grassland::rhi
