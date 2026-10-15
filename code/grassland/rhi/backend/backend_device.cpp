#include "grassland/rhi/backend/backend_device.h"

#include <cassert>

#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_window.h"

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

rhi::IWindow *Device::CreateGLFWWindow(const WindowDesc &desc) {
  try {
    return ManageWindow(CreateNativeWindow(desc)).Detach();
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Window creation ran out of memory.");
    return nullptr;
  }
}

rhi::IQueue *Device::GetQueue(QueueType type) {
  try {
    return CreateQueue(type).release();
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Queue creation ran out of memory.");
    return nullptr;
  }
}

rhi::ICommandContext *Device::CreateCommandContext(rhi::IQueue *queue) {
  const auto impl = dynamic_cast<Queue *>(queue);
  if (!impl || impl->Owner() != this) {
    ReportError(-1, "The queue belongs to another device or implementation.");
    return nullptr;
  }
  try {
    return impl->CreateCommandContext().release();
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Command context allocation failed.");
    return nullptr;
  }
}

const PhysicalDevice &Device::GetPhysicalDevice() const noexcept {
  return info_;
}

Core *Device::OwnerCore() const {
  return core_;
}

ObjectPtr<Window> Device::ManageWindow(std::unique_ptr<Window> window) {
  if (!window)
    return {};
  CollectWindows();
  auto result = ObjectPtr<Window>::Adopt(window.release());
  std::lock_guard<std::mutex> lock(windows_mutex_);
  windows_.push_back(result);
  return result;
}

void Device::CollectWindows() {
  std::lock_guard<std::mutex> lock(windows_mutex_);
  for (auto it = windows_.begin(); it != windows_.end();) {
    if ((*it)->IsUniquelyOwned() && !(*it)->HasFrameLeases())
      it = windows_.erase(it);
    else
      ++it;
  }
}

const void *Device::NativeIdentity() const {
  return this;
}

std::unique_ptr<Window> Device::CreateNativeWindow(const WindowDesc &) {
  ReportError(-1, "Window creation is unavailable.");
  return {};
}

std::unique_ptr<Queue> Device::CreateQueue(QueueType) {
  return {};
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
  changed_.wait(lock, [&] { return status_ != SubmissionStatus::Pending; });
  return status_ == SubmissionStatus::Completed;
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
  if (!commands->CanSubmit() || commands->DeviceIdentity() != NativeIdentity() ||
      (commands->Frame() && !commands->Frame()->IsValid())) {
    ReportError(-1, "Submission requires fresh finished commands and a valid frame from this device.");
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
  if (result == CommandContext::SubmitResult::Rejected) {
    pending_.pop_back();
    return {};
  }
  commands->MarkSubmitted();
  pending_.back().commands = commands;
  changed_.notify_one();
  return result == CommandContext::SubmitResult::Submitted ? completion : ObjectPtr<SubmissionState>{};
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
    } while (status == SubmissionStatus::Pending);
    // Release payload before publishing completion; tokens never keep it alive.
    pending.commands.Reset();
    CollectWindows();
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
  // Derived destructors call this before destroying their native device.
  for (const auto &window : windows_) {
    assert(window->IsUniquelyOwned() && !window->HasFrameLeases());
  }
  windows_.clear();
}

}  // namespace grassland::rhi::backend
