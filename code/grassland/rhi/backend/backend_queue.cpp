#include "grassland/rhi/backend/backend_queue.h"

#include <algorithm>

#include "grassland/rhi/backend/backend_queue_state.h"

namespace grassland::rhi::backend {

ObjectPtr<IBackendSubmissionToken> IBackendQueue::Submit(const ObjectPtr<IBackendCommandContext> &commands,
                                                         const SubmitInfo &info) {
  if (!commands->Seal())
    return {};
  SubmitInfo native_info;
  native_info.waits.reserve(info.waits.size());
  for (const auto &wait : info.waits) {
    auto timeline = rhi::QueryInterfacePtr<IBackendTimeline>(wait.timeline).Get();
    if (!timeline || timeline->Owner() != StateQueue().device_) {
      ReportError(-1, "Submission waits require a timeline from this device.");
      return {};
    }
    // A submit waits for all dependencies; Vulkan requires unique semaphore handles.
    auto existing = std::find_if(native_info.waits.begin(), native_info.waits.end(),
                                 [&](const Event &event) { return event.timeline == wait.timeline; });
    if (existing == native_info.waits.end())
      native_info.waits.push_back(wait);
    else
      existing->value = std::max(existing->value, wait.value);
  }
  std::unique_lock<std::mutex> queue_lock(*StateQueue().mutex_);
  const Event signal{GetTimeline(), NextSerial()};
  auto completion = ObjectPtr<IBackendSubmissionToken>::Adopt(IBackendSubmissionToken::Create(signal));
  // Register before native submission, while the collector cannot take the list.
  auto entry = StateQueue().pending_.insert(StateQueue().pending_.end(), {commands, completion});
  IBackendCommandContext::SubmitResult result;
  try {
    commands->BindDownloads(completion.Get());
    result = commands->SubmitNative(native_info, signal);
    if (result == IBackendCommandContext::SUBMIT_RESULT_REJECTED) {
      StateQueue().pending_.erase(entry);
      completion->Complete(SUBMISSION_STATUS_FAILED);
      commands->ReleaseDownloads();
      return {};
    }
    commands->MarkSubmitted();
  } catch (...) {
    if (!commands->WasSubmitted()) {
      StateQueue().pending_.erase(entry);
      completion->Complete(SUBMISSION_STATUS_FAILED);
      commands->ReleaseDownloads();
    }
    throw;
  }
  queue_lock.unlock();
  // The caller's context reference keeps it alive while the collector retires it.
  // Next-frame preparation belongs to the Window, not this submission's result.
  if (result == IBackendCommandContext::SUBMIT_RESULT_SUBMITTED)
    commands->UpdatePresentedImages();
  return result == IBackendCommandContext::SUBMIT_RESULT_SUBMITTED ? completion : ObjectPtr<IBackendSubmissionToken>{};
}

bool IBackendQueue::CollectCompleted(uint64_t completed, bool timeline_valid) {
  std::list<QueueState::Submission> work;
  {
    std::lock_guard<std::mutex> lock(*StateQueue().mutex_);
    work.splice(work.end(), StateQueue().pending_);
  }
  // Polling may submit a recovery signal; releasing resources may acquire other
  // backend locks. Neither operation runs under the submission mutex.
  for (auto it = work.begin(); it != work.end();) {
    const auto status = it->commands->PollCompletion(it->completion->GetEvent(), completed, timeline_valid);
    if (status == SUBMISSION_STATUS_PENDING) {
      ++it;
      continue;
    }
    it->commands->ReleaseDownloads();
    it->commands.Reset();
    it->completion->Complete(status);
    it = work.erase(it);
  }
  std::lock_guard<std::mutex> lock(*StateQueue().mutex_);
  StateQueue().pending_.splice(StateQueue().pending_.begin(), work);
  return StateQueue().pending_.empty();
}

void IBackendQueue::InitializeQueue(IBackendDevice *device, QueueType type, std::shared_ptr<std::mutex> mutex) {
  StateQueue().device_ = device;
  StateQueue().type_ = type;
  StateQueue().mutex_ = std::move(mutex);
}

std::mutex &IBackendQueue::QueueMutex() {
  return *StateQueue().mutex_;
}

uint64_t IBackendQueue::NextSerial() {
  return ++StateQueue().serial_;
}

IBackendDevice *IBackendQueue::Owner() const {
  return StateQueue().device_;
}

QueueType IBackendQueue::CommonGetType() const {
  return StateQueue().type_;
}

}  // namespace grassland::rhi::backend

LONGMARCH_RHI_UUID(grassland::rhi::backend::IBackendQueue, "5b1b277a-1b68-4947-bf02-14449939898d")

namespace grassland::rhi::backend {
QueueState::QueueState() = default;
QueueState::~QueueState() = default;
}  // namespace grassland::rhi::backend
