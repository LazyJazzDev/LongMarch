#include "grassland/rhi/backend/vulkan/vulkan_queue.h"

#include <algorithm>

#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/vulkan/vulkan_command_context.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_queue_impl.h"

namespace grassland::rhi::backend::vulkan {

rhi::ICommandContext *VulkanQueueImpl::CreateCommandContext() {
  return IVulkanCommandContext::Create(this);
}

IVulkanQueue *VulkanQueueImpl::Create(IVulkanDevice *device,
                                      QueueType type,
                                      VkQueue queue,
                                      std::shared_ptr<std::mutex> mutex) {
  auto result = std::unique_ptr<VulkanQueueImpl>(new VulkanQueueImpl(device, type, queue, std::move(mutex)));
  result->timeline_ = IVulkanTimeline::Create(device);
  if (!result->timeline_)
    return nullptr;
  return result.release();
}

VkQueue VulkanQueueImpl::NativeQueue() const {
  return queue_;
}

IVulkanTimeline *VulkanQueueImpl::GetTimeline() const {
  return timeline_.Get();
}

VkResult VulkanQueueImpl::Wait(uint64_t value, uint64_t timeout) const {
  const auto semaphore = timeline_->NativeSemaphore();
  VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  info.semaphoreCount = 1;
  info.pSemaphores = &semaphore;
  info.pValues = &value;
  return vkWaitSemaphores(rhi::QueryInterfacePtr<IVulkanDevice>(Owner()).Get()->NativeDevice(), &info, timeout);
}

VulkanQueueImpl::VulkanQueueImpl(IVulkanDevice *device,
                                 QueueType type,
                                 VkQueue queue,
                                 std::shared_ptr<std::mutex> mutex)
    : queue_(queue) {
  InitializeQueue(device, type, std::move(mutex));
}

IVulkanQueue *IVulkanQueue::Create(IVulkanDevice *device,
                                   QueueType type,
                                   VkQueue queue,
                                   std::shared_ptr<std::mutex> mutex) {
  return VulkanQueueImpl::Create(device, type, queue, mutex);
}

ObjectPtr<IBackendSubmissionToken> VulkanQueueImpl::Submit(const ObjectPtr<IBackendCommandContext> &commands,
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
  const Event signal{GetTimeline(), NextSerial()};
  auto completion = ObjectPtr<IBackendSubmissionToken>::Adopt(IBackendSubmissionToken::Create(signal));
  // Register before native submission, while the collector cannot take the list.
  auto entry = pending_.insert(pending_.end(), {commands, completion});
  IBackendCommandContext::SubmitResult result;
  try {
    commands->BindDownloads(completion.Get());
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

bool VulkanQueueImpl::CollectCompleted(uint64_t completed, bool timeline_valid) {
  std::list<Submission> work;
  {
    std::lock_guard<std::mutex> lock(*mutex_);
    work.splice(work.end(), pending_);
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
  std::lock_guard<std::mutex> lock(*mutex_);
  pending_.splice(pending_.begin(), work);
  return pending_.empty();
}

void VulkanQueueImpl::InitializeQueue(IBackendDevice *device, QueueType type, std::shared_ptr<std::mutex> mutex) {
  device_ = device;
  type_ = type;
  mutex_ = std::move(mutex);
}

std::mutex &VulkanQueueImpl::QueueMutex() {
  return *mutex_;
}

uint64_t VulkanQueueImpl::NextSerial() {
  return ++serial_;
}

IBackendDevice *VulkanQueueImpl::Owner() const {
  return device_;
}

QueueType VulkanQueueImpl::GetType() const {
  return type_;
}
}  // namespace grassland::rhi::backend::vulkan

LONGMARCH_RHI_UUID(grassland::rhi::backend::vulkan::IVulkanQueue, "09666973-0b3a-4165-bb7f-0faf38e0f01a")
