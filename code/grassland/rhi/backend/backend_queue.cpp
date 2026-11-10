#include "grassland/rhi/backend/backend_queue.h"

#include <algorithm>

namespace grassland::rhi::backend {

ObjectPtr<SubmissionToken> Queue::Submit(const ObjectPtr<CommandContext> &commands, const SubmitInfo &info) {
  if (!commands->Seal())
    return {};
  SubmitInfo native_info;
  native_info.waits.reserve(info.waits.size());
  for (const auto &wait : info.waits) {
#if !defined(NDEBUG)
    auto timeline = dynamic_cast<Timeline *>(wait.timeline);
    if (!timeline || timeline->Owner() != device_) {
      ReportError(-1, "Submission waits require a timeline from this device.");
      return {};
    }
#endif
    // A submit waits for all dependencies; Vulkan requires unique semaphore handles.
    auto existing = std::find_if(native_info.waits.begin(), native_info.waits.end(),
                                 [&](const Event &event) { return event.timeline == wait.timeline; });
    if (existing == native_info.waits.end())
      native_info.waits.push_back(wait);
    else
      existing->value = std::max(existing->value, wait.value);
  }
  std::unique_lock<std::mutex> queue_lock(*mutex_);
  const Event signal{GetTimeline(), NextSerial()};
  auto completion = ObjectPtr<SubmissionToken>::Adopt(new SubmissionToken(signal));
  // Allocate the node before accepting GPU work. Splicing into the collector
  // cannot allocate, including when error reporting throws after native submit.
  std::list<Submission> submission;
  submission.push_back({commands, completion});
  auto publish = [&] {
    if (!queue_lock.owns_lock())
      queue_lock.lock();
    pending_.splice(pending_.end(), submission);
  };
  commands->BindDownloads(completion.Get());
  CommandContext::SubmitResult result;
  try {
    result = commands->SubmitNative(native_info, signal);
    if (result == CommandContext::SUBMIT_RESULT_REJECTED) {
      completion->Complete(SUBMISSION_STATUS_FAILED);
      commands->ReleaseDownloads();
      return {};
    }
    commands->MarkSubmitted();
    queue_lock.unlock();
    if (result == CommandContext::SUBMIT_RESULT_SUBMITTED && !commands->UpdatePresentedImages())
      result = CommandContext::SUBMIT_RESULT_FAILED_IN_FLIGHT;
  } catch (...) {
    if (commands->WasSubmitted())
      publish();
    else {
      completion->Complete(SUBMISSION_STATUS_FAILED);
      commands->ReleaseDownloads();
    }
    throw;
  }
  publish();
  return result == CommandContext::SUBMIT_RESULT_SUBMITTED ? completion : ObjectPtr<SubmissionToken>{};
}

bool Queue::CollectCompleted() {
  std::list<Submission> work;
  {
    std::lock_guard<std::mutex> lock(*mutex_);
    work.splice(work.end(), pending_);
  }
  // Polling may submit a recovery signal; releasing resources may acquire other
  // backend locks. Neither operation runs under the submission mutex.
  for (auto it = work.begin(); it != work.end();) {
    const auto status = it->commands->PollCompletion(it->completion->GetEvent());
    if (status == SUBMISSION_STATUS_PENDING) {
      ++it;
      continue;
    }
    it->commands->ReleaseDownloads();
    it->commands.Reset();
    it->completion->Complete(status);
    it = work.erase(it);
  }
  std::lock_guard<std::mutex> lock(*mutex_);
  pending_.splice(pending_.begin(), work);
  return pending_.empty();
}

Queue::Queue(Device *device, QueueType type, std::shared_ptr<std::mutex> mutex)
    : device_(device),
      type_(type),
      mutex_(std::move(mutex)) {
}

Queue::~Queue() = default;

std::mutex &Queue::QueueMutex() {
  return *mutex_;
}

uint64_t Queue::NextSerial() {
  return ++serial_;
}

Device *Queue::Owner() const {
  return device_;
}

QueueType Queue::GetType() const {
  return type_;
}

}  // namespace grassland::rhi::backend
