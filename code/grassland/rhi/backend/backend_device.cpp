#include "grassland/rhi/backend/backend_device.h"

#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_core.h"

namespace grassland::rhi::backend {
Device::Device(PhysicalDevice info, Core *core) : core_(core), info_(std::move(info)) {
}

Device::~Device() {
  ShutdownQueues();
}

std::string Device::DeviceName() const {
  return info_.name;
}

bool Device::HasRayTracingPipeline() const {
  return info_.ray_tracing_pipeline_feature;
}

bool Device::HasRayQuery() const {
  return info_.ray_query_feature;
}

bool Device::HasGeometryShader() const {
  return info_.geometry_shader_feature;
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

ObjectPtr<SubmissionToken> Device::Submit(const ObjectPtr<CommandContext> &commands,
                                          const std::vector<ObjectPtr<SubmissionToken>> &waits,
                                          ObjectPtr<SubmissionToken> completion) {
  if (!commands->Seal())
    return {};
#if !defined(NDEBUG)
  if (commands->DeviceIdentity() != NativeIdentity()) {
    ReportError(-1, "Submission requires a context from this device.");
    return {};
  }
#endif
  for (const auto &wait : waits) {
#if !defined(NDEBUG)
    if (!wait || wait->Owner() != this) {
      ReportError(-1, "Submission dependency is empty or belongs to another device.");
      return {};
    }
#endif
    if (wait->GetStatus() == SUBMISSION_STATUS_FAILED || wait->GetStatus() == SUBMISSION_STATUS_CANCELED) {
      ReportError(-1, "Submission dependency failed or was canceled.");
      return {};
    }
  }
  // Both logical roles share one physical queue. Its ordering already supplies
  // these GPU dependencies; no host wait or additional native wait is required.
  std::lock_guard<std::mutex> lock(mutex_);
  if (stopping_)
    return {};
  if (!collector_.joinable())
    collector_ = std::thread([this] { Collect(); });
  // Allocate the retirement record before submitting any GPU work.
  pending_.push_back({{}, completion});
  commands->BindDownloads(completion.Get());
  CommandContext::SubmitResult result;
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
      completion->Complete(SUBMISSION_STATUS_FAILED);
      commands->ReleaseDownloads();
    }
    throw;
  }
  if (result == CommandContext::SUBMIT_RESULT_REJECTED) {
    pending_.pop_back();
    completion->Complete(SUBMISSION_STATUS_FAILED);
    commands->ReleaseDownloads();
    return {};
  }
  commands->MarkSubmitted();
  pending_.back().commands = commands;
  changed_.notify_one();
  return result == CommandContext::SUBMIT_RESULT_SUBMITTED ? completion : ObjectPtr<SubmissionToken>{};
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
    pending.commands->ReleaseDownloads();
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
