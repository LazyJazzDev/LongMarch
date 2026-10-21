#include "grassland/rhi/backend/backend_device.h"

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/backend_shader.h"

namespace grassland::rhi::backend {

Device::Device(PhysicalDevice info, Core *core) : core_(core), info_(std::move(info)) {
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
  if (desc.source.empty() || desc.entry_point.empty())
    return nullptr;
  return new Shader(desc);
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
                                          const std::vector<ObjectPtr<SubmissionToken>> &waits) {
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
  auto completion = ObjectPtr<SubmissionToken>::Adopt(new SubmissionToken(this));
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
  if (result == CommandContext::SUBMIT_RESULT_REJECTED) {
    pending_.pop_back();
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
