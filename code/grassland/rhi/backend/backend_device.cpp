#include "grassland/rhi/backend/backend_device.h"

#include <cassert>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_core.h"
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
  collector_points_ = {{nullptr, 1}, {nullptr, 1}};
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
  if (type != QUEUE_TYPE_GRAPHICS && type != QUEUE_TYPE_TRANSFER)
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

ObjectPtr<SubmissionToken> Device::Submit(const ObjectPtr<CommandContext> &commands,
                                          const std::vector<Event> &waits,
                                          Queue *queue) {
  if (!commands->Seal())
    return {};
#if !defined(NDEBUG)
  if (commands->DeviceIdentity() != NativeIdentity()) {
    ReportError(-1, "Submission requires a context from this device.");
    return {};
  }
#endif
  for (const auto &wait : waits) {
    const auto timeline = dynamic_cast<Timeline *>(wait.timeline);
    if (!timeline || timeline->Owner() != this) {
      ReportError(-1, "Submission dependency is empty or belongs to another device.");
      return {};
    }
    const auto status = wait.GetStatus();
    if (status == SUBMISSION_STATUS_FAILED || status == SUBMISSION_STATUS_CANCELED) {
      ReportError(-1, "Submission dependency failed or was canceled.");
      return {};
    }
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (stopping_)
    return {};
  std::unique_lock<std::mutex> queue_lock(queue->Mutex());
  const Event event{queue->TimelineState(), queue->NextSerial()};
  auto completion = ObjectPtr<SubmissionToken>::Adopt(new SubmissionToken(this, event));
  if (!collector_.joinable())
    collector_ = std::thread([this] { Collect(); });
  // Allocate the retirement record before submitting any GPU work.
  pending_.push_back({{}, completion});
  auto &submission = pending_.back();
  commands->BindDownloads(completion.Get());
  CommandContext::SubmitResult result;
  try {
    result = commands->SubmitNative(waits, submission);
    queue_lock.unlock();
    if (result == CommandContext::SUBMIT_RESULT_SUBMITTED)
      result = commands->FinishSubmitNative(submission);
  } catch (...) {
    if (queue_lock.owns_lock())
      queue_lock.unlock();
    // Even error reporting can allocate after a native submit. Keep already
    // submitted work alive before propagating to the public allocation boundary.
    if (commands->WasSubmitted()) {
      pending_.back().commands = commands;
      if (HasNativeTimelines())
        WakeCollector();
      if (!HasNativeTimelines())
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
  if (!HasNativeTimelines())
    changed_.notify_one();
  if (result == CommandContext::SUBMIT_RESULT_FAILED_IN_FLIGHT)
    WakeCollector();
  return result == CommandContext::SUBMIT_RESULT_SUBMITTED ? completion : ObjectPtr<SubmissionToken>{};
}

void Device::Collect() {
  if (HasNativeTimelines()) {
    CollectTimelines();
    return;
  }
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
      status = pending.commands->WaitForProgress(pending);
    } while (status == SUBMISSION_STATUS_PENDING);
    // Release payload before publishing completion; tokens never keep it alive.
    pending.commands->ReleaseDownloads();
    pending.commands.Reset();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto event = pending.completion->GetEvent();
      static_cast<Timeline *>(event.timeline)->RecordCompletion(event.value);
    }
    changed_.notify_all();
    pending.completion->Complete(status);
  }
}

void Device::ShutdownQueues() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  if (collector_.joinable())
    WakeCollector();
  changed_.notify_all();
  if (collector_.joinable())
    collector_.join();
}

bool Device::HasNativeTimelines() const {
  return queues_[0] && queues_[0]->TimelineState()->IsNative();
}

bool Device::WaitTimelines(const std::vector<Event> &points, bool wait_all) const {
  std::unique_lock<std::mutex> lock(mutex_);
  changed_.wait(lock, [&] {
    bool all = true, any = false;
    for (auto p : points) {
      bool done = [&] {
        uint64_t value{};
        p.timeline->GetCompletedValue(value);
        return value >= p.value;
      }();
      all &= done;
      any |= done;
    }
    return wait_all ? all : any;
  });
  return true;
}

void Device::WakeCollector() {
}

bool Device::WaitEvents(const std::vector<Event> &events, bool wait_all) const {
  if (events.empty())
    return false;
  for (auto event : events) {
    auto timeline = dynamic_cast<Timeline *>(event.timeline);
    if (!timeline || timeline->Owner() != this)
      return false;
  }
  return WaitTimelines(events, wait_all);
}

void Device::CollectTimelines() {
  for (;;) {
    std::array<uint64_t, 2> values{};
    bool valid = true;
    for (int i = 0; i < 2; ++i)
      valid &= collector_points_[i].timeline->GetCompletedValue(values[i]);
    std::list<Submission> ready;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto it = pending_.begin(); it != pending_.end();) {
        const auto event = it->completion->GetEvent();
        bool completed = !valid;
        for (size_t i = 0; i < collector_points_.size(); ++i)
          if (event.timeline == collector_points_[i].timeline)
            completed |= values[i] >= event.value;
        if (completed) {
          auto completed = it++;
          ready.splice(ready.end(), pending_, completed);
        } else
          ++it;
      }
      if (ready.empty() && stopping_ && pending_.empty())
        return;
    }
    for (auto &entry : ready) {
      auto status = entry.commands->WaitForProgress(entry);
      while (status == SUBMISSION_STATUS_PENDING)
        status = entry.commands->WaitForProgress(entry);
      entry.commands->ReleaseDownloads();
      entry.commands.Reset();
      entry.completion->Complete(status);
    }
    if (!ready.empty())
      continue;
    if (!valid) {
      std::unique_lock<std::mutex> lock(mutex_);
      changed_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
      continue;
    }
    // Queue identities are fixed. Wait for progress even before a new Submit exists.
    collector_points_[0].value = values[0] + 1;
    collector_points_[1].value = values[1] + 1;
    if (!WaitTimelines(collector_points_, false))
      std::this_thread::yield();
  }
}

Queue *Device::QueueState(QueueType type) const {
  return queues_[type].Get();
}

void Device::InitializeQueue(ObjectPtr<Queue> queue) {
  assert(queue && queue->Owner() == this && !collector_.joinable() && pending_.empty());
  const auto type = queue->GetType();
  assert(!queues_[type]);
  queues_[type] = std::move(queue);
  collector_points_[type].timeline = queues_[type]->TimelineState();
}

void Device::ReleaseQueues() {
  for (auto &queue : queues_)
    queue.Reset();
}
}  // namespace grassland::rhi::backend
