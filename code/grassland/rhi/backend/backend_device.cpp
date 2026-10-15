#include "grassland/rhi/backend/backend_device.h"

#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_core.h"

namespace grassland::rhi::backend {

namespace {
std::atomic<uint64_t> next_identity{1};
}

Device::Device(PhysicalDevice info, Core *core)
    : core_(core),
      info_(std::move(info)),
      queue_identity_(next_identity.fetch_add(1)) {
}

Device::~Device() {
  ShutdownQueues();
}

std::string Device::Name() const {
  return info_.name;
}

bool Device::SupportRayTracingPipeline() const {
  return info_.ray_tracing_pipeline_feature;
}

bool Device::SupportRayQuery() const {
  return info_.ray_query_feature;
}

bool Device::SupportGeometryShader() const {
  return info_.geometry_shader_feature;
}

rhi::IWindow *Device::CreateGLFWWindow(const WindowDesc &) {
  ReportError(-1, "Window creation is unavailable.");
  return nullptr;
}

rhi::IQueue *Device::GetQueue(QueueType) {
  return nullptr;
}

rhi::ICommandContext *Device::CreateCommandContext() {
  return nullptr;
}

const PhysicalDevice &Device::GetPhysicalDevice() const noexcept {
  return info_;
}

Core *Device::OwnerCore() const {
  return core_;
}

const void *Device::NativeIdentity() const {
  return this;
}

Device::SubmissionState::SubmissionState(uint64_t identity) : queue_identity_(identity) {
}

uint64_t Device::SubmissionState::QueueIdentity() const {
  return queue_identity_;
}

SubmissionStatus Device::SubmissionState::GetStatus() const {
  return Status();
}

bool Device::SubmissionState::WaitHost() const {
  return Wait();
}

SubmissionStatus Device::SubmissionState::Status() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return status_;
}

bool Device::SubmissionState::Wait() const {
  std::unique_lock<std::mutex> lock(mutex_);
  changed_.wait(lock, [&] { return status_ != SUBMISSION_STATUS_PENDING; });
  return status_ == SUBMISSION_STATUS_COMPLETED;
}

void Device::SubmissionState::Complete(SubmissionStatus status) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    status_ = status;
  }
  changed_.notify_all();
}

ObjectPtr<Device::SubmissionState> Device::Submit(const ObjectPtr<CommandContext> &commands,
                                                  const std::vector<ObjectPtr<Device::SubmissionState>> &waits) {
  if (!commands->Seal())
    return {};
  if (commands->DeviceIdentity() != NativeIdentity() || (commands->Frame() && !commands->Frame()->IsValid())) {
    ReportError(-1, "Submission requires a fresh context and a valid frame from this device.");
    return {};
  }
  for (const auto &wait : waits) {
    if (!wait || wait->QueueIdentity() != queue_identity_ || wait->Status() == SUBMISSION_STATUS_FAILED ||
        wait->Status() == SUBMISSION_STATUS_CANCELED) {
      ReportError(-1, "Submission dependency is empty, failed, or belongs to another device.");
      return {};
    }
  }
  // Both logical roles share one physical queue. Its ordering already supplies
  // these GPU dependencies; no host wait or additional native wait is required.
  auto completion = ObjectPtr<Device::SubmissionState>::Adopt(new Device::SubmissionState(queue_identity_));
  std::lock_guard<std::mutex> lock(mutex_);
  if (stopping_)
    return {};
  if (!collector_.joinable())
    collector_ = std::thread([this] { Collect(); });
  // Allocate the retirement record before submitting any GPU work.
  pending_.push_back({{}, completion});
  CommandContext::SubmitResult result;
  try {
    result = commands->SubmitNative();
    if (result == CommandContext::SUBMIT_RESULT_SUBMITTED)
      commands->AdvanceFrame();
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
  if (result == CommandContext::SUBMIT_RESULT_REJECTED) {
    pending_.pop_back();
    return {};
  }
  commands->MarkSubmitted();
  pending_.back().commands = commands;
  changed_.notify_one();
  return result == CommandContext::SUBMIT_RESULT_SUBMITTED ? completion : ObjectPtr<SubmissionState>{};
}

void Device::Collect() {
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
    } while (status == SUBMISSION_STATUS_PENDING);
    // Release payload before publishing completion; tokens never keep it alive.
    pending.commands.Reset();
    pending.completion->Complete(status);
  }
}

void Device::ShutdownQueues() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  changed_.notify_one();
  if (collector_.joinable())
    collector_.join();
}

}  // namespace grassland::rhi::backend
