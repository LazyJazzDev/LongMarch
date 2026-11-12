#include "grassland/rhi/backend/metal/metal_queue.h"

#include <algorithm>
#include <list>
#include <mutex>

#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/metal/metal_command_context.h"
#include "grassland/rhi/backend/metal/metal_diagnostics.h"
#include "grassland/rhi/backend/metal/metal_queue_impl.h"
#include "grassland/rhi/rhi_queue.h"

namespace grassland::rhi::backend::metal {

rhi::ICommandContext *MetalQueueImpl::CreateCommandContext() {
  return CreateMetalCommandContext(this);
}

IMetalQueue *MetalQueueImpl::Create(IMetalDevice *device, QueueType type, MTL::CommandBufferDescriptor *descriptor) {
  auto result = std::unique_ptr<MetalQueueImpl>(new MetalQueueImpl(device, type));
  result->queue_ = NS::TransferPtr(device->NativeDevice()->newCommandQueue());
  if (!result->queue_) {
    ReportError(-1, "Metal command queue creation failed.");
    return nullptr;
  }
  result->descriptor_ = NS::RetainPtr(descriptor);
  result->timeline_ = IMetalTimeline::Create(device);
  if (!result->timeline_)
    return nullptr;
  return result.release();
}

IMetalTimeline *MetalQueueImpl::GetTimeline() const {
  return timeline_.Get();
}

MTL::CommandQueue *MetalQueueImpl::NativeQueue() const {
  return queue_.get();
}

NS::SharedPtr<MTL::CommandBuffer> MetalQueueImpl::CreateCommandBuffer() {
  return NS::RetainPtr(queue_->commandBuffer(descriptor_.get()));
}

bool MetalQueueImpl::WaitIdle() {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto marker = CreateCommandBuffer();
  if (!marker) {
    ReportError(-1, "Metal idle marker allocation failed.");
    return false;
  }
  marker->commit();
  marker->waitUntilCompleted();
  if (marker->status() != MTL::CommandBufferStatusCompleted) {
    ReportMetalError(marker->error(), "Metal idle marker failed",
                     descriptor_->errorOptions() != MTL::CommandBufferErrorOptionNone);
    return false;
  }
  return true;
}

MetalQueueImpl::MetalQueueImpl(IMetalDevice *device, QueueType type) {
  InitializeQueue(device, type);
}

IMetalQueue *IMetalQueue::Create(IMetalDevice *device, QueueType type, MTL::CommandBufferDescriptor *descriptor) {
  return MetalQueueImpl::Create(device, type, descriptor);
}

ObjectPtr<IBackendSubmissionToken> MetalQueueImpl::Submit(const ObjectPtr<IMetalCommandContext> &commands,
                                                          const SubmitInfo &info) {
  if (!commands->Seal())
    return {};
  SubmitInfo native_info;
  native_info.waits.reserve(info.waits.size());
  for (const auto &wait : info.waits) {
    auto timeline = rhi::QueryInterfacePtr<IMetalTimeline>(wait.timeline).Get();
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
  IMetalCommandContext::SubmitResult result;
  try {
    commands->BindDownloads(rhi::QueryInterfacePtr<rhi::ISubmissionToken>(completion.Get()).Get());
    result = commands->SubmitNative(native_info, signal);
    if (result == IMetalCommandContext::SUBMIT_RESULT_REJECTED) {
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
  if (result == IMetalCommandContext::SUBMIT_RESULT_SUBMITTED)
    commands->UpdatePresentedImages();
  return result == IMetalCommandContext::SUBMIT_RESULT_SUBMITTED ? completion : ObjectPtr<IBackendSubmissionToken>{};
}

bool MetalQueueImpl::CollectCompleted(uint64_t completed, bool timeline_valid) {
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

void MetalQueueImpl::InitializeQueue(IMetalDevice *device, QueueType type, std::shared_ptr<std::mutex> mutex) {
  device_ = device;
  type_ = type;
  mutex_ = std::move(mutex);
}

std::mutex &MetalQueueImpl::QueueMutex() {
  return *mutex_;
}

uint64_t MetalQueueImpl::NextSerial() {
  return ++serial_;
}

IMetalDevice *MetalQueueImpl::Owner() const {
  return device_;
}

QueueType MetalQueueImpl::GetType() const {
  return type_;
}
}  // namespace grassland::rhi::backend::metal
