#include "grassland/rhi/backend/backend_device.h"

#include <algorithm>

#include "grassland/rhi/backend/backend_binding_group.h"
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

#if defined(LONGMARCH_CUDA_RUNTIME)
ICUDABuffer *Device::CreateCUDABuffer(const BufferDesc &) {
  ReportError(-1, "CUDA buffer allocation is unavailable for this backend.");
  return nullptr;
}
#endif

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

int Device::GetCUDADeviceIndex() const noexcept {
  return info_.cuda_device;
}

const PhysicalDevice &Device::GetPhysicalDevice() const noexcept {
  return info_;
}

Core *Device::OwnerCore() const {
  return core_;
}

void Device::StartCollector() {
  collector_ = std::thread([this] { Collect(); });
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
  std::vector<Event> next;
  next.reserve(queues_.size());
  for (;;) {
    next.clear();
    bool empty = true;
    for (const auto &queue : queues_) {
      if (!queue)
        continue;
      uint64_t completed = 0;
      const bool timeline_valid = queue->GetTimeline()->GetCompletedValue(completed);
      if (timeline_valid && completed != UINT64_MAX)
        next.push_back({queue->GetTimeline(), completed + 1});
      if (!queue->CollectCompleted(completed, timeline_valid))
        empty = false;
    }
    if (stopping_ && empty)
      return;
    WaitForProgress(next);
  }
}

void Device::ShutdownQueues() {
  if (!collector_.joinable())
    return;
  stopping_ = true;
  InterruptCollector();
  collector_.join();
}

}  // namespace grassland::rhi::backend
