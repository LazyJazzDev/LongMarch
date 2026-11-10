#include "grassland/rhi/backend/backend_device.h"

#include <algorithm>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_shader.h"

namespace grassland::rhi::backend {
IBottomLevelAccelerationStructure *Device::CreateBottomLevelAccelerationStructure(
    const BottomLevelAccelerationStructureDesc &) {
  ReportError(-1, "Acceleration structures are unavailable for this backend.");
  return nullptr;
}

ITopLevelAccelerationStructure *Device::CreateTopLevelAccelerationStructure(const TopLevelAccelerationStructureDesc &) {
  ReportError(-1, "Acceleration structures are unavailable for this backend.");
  return nullptr;
}

IRayTracingProgram *Device::CreateRayTracingProgram(const RayTracingProgramDesc &) {
  ReportError(-1, "Ray tracing pipelines are unavailable for this backend.");
  return nullptr;
}

Device::Device(PhysicalDevice info, Core *core, BackendAPI api) : api_(api), core_(core), info_(std::move(info)) {
}

rhi::IImage *Device::CreateImage(const ImageDesc &) {
  ReportError(-1, "Independent images are not implemented for this backend.");
  return nullptr;
}

rhi::IBuffer *Device::CreateBuffer(const BufferDesc &) {
  ReportError(-1, "Buffers are not implemented for this backend.");
  return nullptr;
}

rhi::IComputeProgram *Device::CreateComputeProgram(const ComputeProgramDesc &) {
  ReportError(-1, "Compute programs are not implemented for this backend.");
  return nullptr;
}

rhi::IProgram *Device::CreateProgram(const ProgramDesc &) {
  ReportError(-1, "Graphics programs are not implemented for this backend.");
  return nullptr;
}

rhi::IShader *Device::CreateShader(const ShaderDesc &desc) {
  if (desc.filename.empty() || desc.entry_point.empty() || desc.stage < SHADER_STAGE_VERTEX ||
      desc.stage > SHADER_STAGE_CALLABLE)
    return nullptr;
  return Shader::Create(desc, api_);
}

ISampler *Device::CreateSampler(const SamplerDesc &) {
  ReportError(-1, "Samplers are not implemented for this backend.");
  return nullptr;
}

IBindingGroup *Device::CreateBindingGroup(const BindingLayoutDesc &) {
  ReportError(-1, "Binding groups are not implemented for this backend.");
  return nullptr;
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

rhi::IQueue *Device::GetQueue(QueueType type) {
  if (type != QUEUE_TYPE_MAIN && type != QUEUE_TYPE_ASSIST)
    return nullptr;
  return ObjectPtr<Queue>::Retain(queues_[type].Get()).Detach();
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

ObjectPtr<SubmissionToken> Device::Submit(const ObjectPtr<CommandContext> &commands, const SubmitInfo &info) {
  if (!commands->Seal())
    return {};
  SubmitInfo native_info;
  native_info.waits.reserve(info.waits.size());
  for (const auto &wait : info.waits) {
#if !defined(NDEBUG)
    auto timeline = dynamic_cast<Timeline *>(wait.timeline);
    if (!timeline || timeline->Owner() != this) {
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
  auto queue = commands->OwnerQueue();
  std::unique_lock<std::mutex> queue_lock(queue->QueueMutex());
  const Event signal{queue->GetTimeline(), queue->NextSerial()};
  auto completion = ObjectPtr<SubmissionToken>::Adopt(new SubmissionToken(signal));
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
    result = commands->SubmitNative(native_info, signal);
  } catch (...) {
    // Even error reporting can allocate after a native submit. Keep already
    // submitted work alive before propagating to the public allocation boundary.
    if (commands->WasSubmitted()) {
      pending_.back().commands = commands;
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
  queue_lock.unlock();
  if (result == CommandContext::SUBMIT_RESULT_SUBMITTED && !commands->UpdatePresentedImages())
    return {};
  return result == CommandContext::SUBMIT_RESULT_SUBMITTED ? completion : ObjectPtr<SubmissionToken>{};
}

bool Device::WaitEvents(const std::vector<Event> &events, bool wait_all) const {
  if (events.empty())
    return wait_all;
  std::vector<Event> unique;
  unique.reserve(events.size());
  for (const auto &event : events) {
#if !defined(NDEBUG)
    auto timeline = dynamic_cast<Timeline *>(event.timeline);
    if (!timeline || timeline->Owner() != this) {
      ReportError(-1, "Host waits require timelines from this device.");
      return false;
    }
#endif
    auto found = std::find_if(unique.begin(), unique.end(),
                              [&](const Event &other) { return other.timeline == event.timeline; });
    if (found == unique.end())
      unique.push_back(event);
    else
      found->value = wait_all ? std::max(found->value, event.value) : std::min(found->value, event.value);
  }
  return WaitNativeEvents(unique, wait_all);
}

void Device::Collect() {
  std::list<Submission> work;
  std::vector<Event> next;
  next.reserve(queues_.size());
  for (;;) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      work.splice(work.end(), pending_);
      if (stopping_ && work.empty())
        return;
    }
    // Snapshot before polling: progress racing with the poll will satisfy this
    // wait, even when no subsequent submit arrives to wake the collector.
    next.clear();
    for (const auto &queue : queues_) {
      uint64_t completed;
      if (queue && queue->GetTimeline()->GetCompletedValue(completed) && completed != UINT64_MAX)
        next.push_back({queue->GetTimeline(), completed + 1});
    }
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
    // A bounded native wait also permits shutdown and error recovery without
    // adding a synthetic GPU signal to any queue's monotonic timeline.
    if (next.empty())
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    else
      WaitForProgress(next);
  }
}

void Device::ShutdownQueues() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  if (collector_.joinable())
    collector_.join();
}

}  // namespace grassland::rhi::backend
