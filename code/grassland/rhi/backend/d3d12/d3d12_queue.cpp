#include "grassland/rhi/backend/d3d12/d3d12_queue.h"

#include <algorithm>

#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/d3d12/d3d12_command_context.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_queue_impl.h"

namespace grassland::rhi::backend::d3d12 {

rhi::ICommandContext *D3D12QueueImpl::CreateCommandContext() {
  return CreateD3D12CommandContext(this);
}

ID3D12Queue *D3D12QueueImpl::Create(ID3D12Device *device, QueueType type) {
  auto result = std::unique_ptr<D3D12QueueImpl>(new D3D12QueueImpl(device, type));
  D3D12_COMMAND_QUEUE_DESC desc{};
  desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  if (!Check(device->NativeDevice()->CreateCommandQueue(&desc, IID_PPV_ARGS(&result->queue_)), "Create command queue"))
    return nullptr;
  result->timeline_ = ID3D12Timeline::Create(device);
  if (!result->timeline_)
    return nullptr;
  return result.release();
}

ID3D12CommandQueue *D3D12QueueImpl::NativeQueue() const {
  return queue_.Get();
}

ID3D12Timeline *D3D12QueueImpl::GetTimeline() const {
  return timeline_.Get();
}

bool D3D12QueueImpl::WaitIdle() {
  const auto value = NextSerial();
  return Check(queue_->Signal(timeline_->NativeFence(), value), "Signal idle timeline") &&
         rhi::QueryInterfacePtr<ITimeline>(timeline_.Get())->WaitHost(value);
}

D3D12QueueImpl::D3D12QueueImpl(ID3D12Device *device, QueueType type) {
  InitializeQueue(device, type);
}

ID3D12Queue *ID3D12Queue::Create(ID3D12Device *device, QueueType type) {
  return D3D12QueueImpl::Create(device, type);
}

ObjectPtr<IBackendSubmissionToken> D3D12QueueImpl::Submit(const ObjectPtr<IBackendCommandContext> &commands,
                                                          const SubmitInfo &info) {
  if (!commands->Seal())
    return {};
  SubmitInfo native_info;
  native_info.waits.reserve(info.waits.size());
  for (const auto &wait : info.waits) {
    auto timeline = rhi::QueryInterfacePtr<IBackendTimeline>(wait.timeline).Get();
    if (!timeline || timeline->Owner() != device_) {
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
  std::unique_lock<std::mutex> queue_lock(*mutex_);
  const Event signal{rhi::QueryInterfacePtr<ITimeline>(GetTimeline()).Get(), NextSerial()};
  auto completion = ObjectPtr<IBackendSubmissionToken>::Adopt(IBackendSubmissionToken::Create(signal));
  // Register before native submission, while the collector cannot take the list.
  auto entry = pending_.insert(pending_.end(), {commands, completion});
  IBackendCommandContext::SubmitResult result;
  try {
    commands->BindDownloads(rhi::QueryInterfacePtr<rhi::ISubmissionToken>(completion.Get()).Get());
    result = commands->SubmitNative(native_info, signal);
    if (result == IBackendCommandContext::SUBMIT_RESULT_REJECTED) {
      pending_.erase(entry);
      completion->Complete(SUBMISSION_STATUS_FAILED);
      commands->ReleaseDownloads();
      return {};
    }
    commands->MarkSubmitted();
  } catch (...) {
    if (!commands->WasSubmitted()) {
      pending_.erase(entry);
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

bool D3D12QueueImpl::CollectCompleted(uint64_t completed, bool timeline_valid) {
  std::list<Submission> work;
  {
    std::lock_guard<std::mutex> lock(*mutex_);
    work.splice(work.end(), pending_);
  }
  // Polling may submit a recovery signal; releasing resources may acquire other
  // backend locks. Neither operation runs under the submission mutex.
  for (auto it = work.begin(); it != work.end();) {
    const auto status = it->commands->PollCompletion(
        rhi::QueryInterfacePtr<ISubmissionToken>(it->completion.Get())->GetEvent(), completed, timeline_valid);
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

void D3D12QueueImpl::InitializeQueue(ID3D12Device *device, QueueType type, std::shared_ptr<std::mutex> mutex) {
  device_ = device;
  type_ = type;
  mutex_ = std::move(mutex);
}

std::mutex &D3D12QueueImpl::QueueMutex() {
  return *mutex_;
}

uint64_t D3D12QueueImpl::NextSerial() {
  return ++serial_;
}

ID3D12Device *D3D12QueueImpl::Owner() const {
  return device_;
}

QueueType D3D12QueueImpl::GetType() const {
  return type_;
}
}  // namespace grassland::rhi::backend::d3d12
