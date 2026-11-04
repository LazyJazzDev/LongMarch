#include "grassland/rhi/backend/backend_device.h"

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

ObjectPtr<SubmissionToken> Device::Submit(const ObjectPtr<CommandContext> &commands,
                                          const std::vector<ObjectPtr<Event>> &waits,
                                          QueueType queue) {
  if (!commands->SupportsQueue(queue)) {
    ReportError(-1, "The command context or one of its images requires a different queue.");
    return {};
  }
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
    const auto status = wait->GetStatus();
    if (status == SUBMISSION_STATUS_FAILED || status == SUBMISSION_STATUS_CANCELED) {
      ReportError(-1, "Submission dependency failed or was canceled.");
      return {};
    }
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (stopping_)
    return {};
  const TimelinePoint point{queue, ++submitted_[queue]};
  auto event = ObjectPtr<Event>::Adopt(new Event(this, queue));
  auto completion = ObjectPtr<SubmissionToken>::Adopt(new SubmissionToken(this, event));
  commands->ConfigureSubmission(queue, point.value, waits);
  if (!collector_.joinable())
    collector_ = std::thread([this] { Collect(); });
  // Allocate the retirement record before submitting any GPU work.
  pending_.push_back({{}, completion, point});
  commands->BindDownloads(completion.Get());
  CommandContext::SubmitResult result;
  try {
    result = commands->SubmitNative();
  } catch (...) {
    // Even error reporting can allocate after a native submit. Keep already
    // submitted work alive before propagating to the public allocation boundary.
    if (commands->WasSubmitted()) {
      event->BindSignalValue(commands->SubmissionValue());
      pending_.back().point.value = commands->SubmissionValue();
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
  event->BindSignalValue(commands->SubmissionValue());
  pending_.back().point.value = commands->SubmissionValue();
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
      status = pending.commands->WaitForProgress();
    } while (status == SUBMISSION_STATUS_PENDING);
    // Release payload before publishing completion; tokens never keep it alive.
    pending.commands->ReleaseDownloads();
    pending.commands.Reset();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      completed_[pending.point.queue] = pending.point.value;
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

ICommandContext *Device::CreateContextForQueue(QueueType) {
  return CreateCommandContext();
}

bool Device::SharesNativeQueue(QueueType, QueueType) const {
  return true;
}

bool Device::HasNativeTimelines() const {
  return false;
}

bool Device::QueryTimeline(QueueType type, uint64_t &value) const {
  std::lock_guard<std::mutex> lock(mutex_);
  value = completed_[type];
  return true;
}

bool Device::WaitTimelines(const std::vector<TimelinePoint> &points, bool wait_all) const {
  std::unique_lock<std::mutex> lock(mutex_);
  changed_.wait(lock, [&] {
    bool all = true, any = false;
    for (auto p : points) {
      bool done = completed_[p.queue] >= p.value;
      all &= done;
      any |= done;
    }
    return wait_all ? all : any;
  });
  return true;
}

void Device::WakeCollector() {
}

SubmissionStatus Device::QueryEvent(TimelinePoint point) const {
  uint64_t value{};
  if (!QueryTimeline(point.queue, value))
    return SUBMISSION_STATUS_FAILED;
  return value >= point.value ? SUBMISSION_STATUS_COMPLETED : SUBMISSION_STATUS_PENDING;
}

bool Device::WaitEvents(const std::vector<IEvent *> &events, bool wait_all) const {
  if (events.empty())
    return false;
  std::vector<TimelinePoint> points;
  for (auto event : events) {
    auto native = dynamic_cast<Event *>(event);
    if (!native || native->Owner() != this)
      return false;
    points.push_back(native->Point());
  }
  return WaitTimelines(points, wait_all);
}

void Device::CollectTimelines() {
  for (;;) {
    std::array<uint64_t, 2> values{};
    bool valid = true;
    for (int i = 0; i < 2; ++i)
      valid &= QueryTimeline(static_cast<QueueType>(i), values[i]);
    std::list<Submission> ready;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto it = pending_.begin(); it != pending_.end();) {
        if (!valid || values[it->point.queue] >= it->point.value) {
          auto completed = it++;
          ready.splice(ready.end(), pending_, completed);
        } else
          ++it;
      }
      if (ready.empty() && stopping_ && pending_.empty())
        return;
    }
    for (auto &entry : ready) {
      auto status = entry.commands->WaitForProgress();
      while (status == SUBMISSION_STATUS_PENDING)
        status = entry.commands->WaitForProgress();
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

}  // namespace grassland::rhi::backend
