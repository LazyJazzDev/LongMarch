#include "grassland/rhi/backend/backend_device.h"

#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_window.h"

namespace grassland::rhi::backend {

namespace {
std::atomic<uint64_t> next_identity{1};
}

IDevice::IDevice(PhysicalDevice info, const ObjectPtr<ICore> &core)
    : core_(core),
      info_(std::move(info)),
      queue_identity_(next_identity.fetch_add(1)) {
}

IDevice::~IDevice() {
  ShutdownQueues();
}

const PhysicalDevice &IDevice::GetPhysicalDevice() const noexcept {
  return info_;
}

const void *IDevice::NativeIdentity() const {
  return this;
}

std::unique_ptr<IWindow> IDevice::CreateWindow(const WindowDesc &) {
  ReportError(-1, "Window creation is unavailable.");
  return {};
}

std::unique_ptr<IQueue> IDevice::CreateQueue(QueueType) {
  return {};
}

IDevice::SubmissionState::SubmissionState(uint64_t identity) : queue_identity_(identity) {
}

uint64_t IDevice::SubmissionState::QueueIdentity() const {
  return queue_identity_;
}

SubmissionStatus IDevice::SubmissionState::Status() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return status_;
}

bool IDevice::SubmissionState::Wait() const {
  std::unique_lock<std::mutex> lock(mutex_);
  changed_.wait(lock, [&] { return status_ != SubmissionStatus::Pending; });
  return status_ == SubmissionStatus::Completed;
}

void IDevice::SubmissionState::Complete(SubmissionStatus status) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    status_ = status;
  }
  changed_.notify_all();
}

ObjectPtr<IDevice::SubmissionState> IDevice::Submit(const ObjectPtr<ICommandContext> &commands,
                                                    const std::vector<ObjectPtr<IDevice::SubmissionState>> &waits) {
  if (commands->DeviceIdentity() != NativeIdentity() || (commands->Frame() && !commands->Frame()->IsValid())) {
    ReportError(-1, "Submission uses another device or an invalidated swapchain frame.");
    return {};
  }
  for (const auto &wait : waits) {
    if (!wait || wait->QueueIdentity() != queue_identity_ || wait->Status() == SubmissionStatus::Failed ||
        wait->Status() == SubmissionStatus::Canceled) {
      ReportError(-1, "Submission dependency is empty, failed, or belongs to another device.");
      return {};
    }
  }
  // Both logical roles share one physical queue. Its ordering already supplies
  // these GPU dependencies; no host wait or additional native wait is required.
  auto completion = ObjectPtr<IDevice::SubmissionState>::Adopt(new IDevice::SubmissionState(queue_identity_));
  std::lock_guard<std::mutex> lock(mutex_);
  if (stopping_)
    return {};
  if (!collector_.joinable())
    collector_ = std::thread([this] { Collect(); });
  // Allocate the retirement record before submitting any GPU work.
  pending_.push_back({{}, completion});
  ICommandContext::SubmitResult result;
  try {
    result = commands->SubmitNative();
  } catch (...) {
    // Even error reporting can allocate after a native submit. Keep already
    // submitted work alive before propagating to the public allocation boundary.
    if (commands->WasSubmitted()) {
      pending_.back().commands = commands;
      changed_.notify_one();
    } else {
      pending_.pop_back();
    }
    throw;
  }
  if (result == ICommandContext::SubmitResult::Rejected) {
    pending_.pop_back();
    return {};
  }
  commands->MarkSubmitted();
  pending_.back().commands = commands;
  changed_.notify_one();
  return result == ICommandContext::SubmitResult::Submitted ? completion : ObjectPtr<SubmissionState>{};
}

void IDevice::Collect() {
  for (;;) {
    Submission pending;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      changed_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
      if (pending_.empty())
        return;
      pending = std::move(pending_.front());
      pending_.pop_front();
    }
    SubmissionStatus status;
    do {
      status = pending.commands->WaitForProgress();
    } while (status == SubmissionStatus::Pending);
    // Release payload before publishing completion; tokens never keep it alive.
    pending.commands.Reset();
    pending.completion->Complete(status);
  }
}

void IDevice::ShutdownQueues() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  changed_.notify_one();
  if (collector_.joinable())
    collector_.join();
}

}  // namespace grassland::rhi::backend
