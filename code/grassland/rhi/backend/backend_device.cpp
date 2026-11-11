#include "grassland/rhi/backend/backend_device.h"

#include <algorithm>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/backend_device_state.h"
#include "grassland/rhi/backend/backend_queue.h"
#include "grassland/rhi/backend/backend_shader.h"
#include "grassland/rhi/backend/backend_shader_impl.h"

namespace grassland::rhi::backend {
IBottomLevelAccelerationStructure *IBackendDevice::CommonCreateBottomLevelAccelerationStructure(
    const BottomLevelAccelerationStructureDesc &) {
  ReportError(-1, "Acceleration structures are unavailable for this backend.");
  return nullptr;
}

ITopLevelAccelerationStructure *IBackendDevice::CommonCreateTopLevelAccelerationStructure(
    const TopLevelAccelerationStructureDesc &) {
  ReportError(-1, "Acceleration structures are unavailable for this backend.");
  return nullptr;
}

IRayTracingProgram *IBackendDevice::CommonCreateRayTracingProgram(const RayTracingProgramDesc &) {
  ReportError(-1, "Ray tracing pipelines are unavailable for this backend.");
  return nullptr;
}

void IBackendDevice::InitializeDevice(PhysicalDevice info, IBackendCore *core, BackendAPI api) {
  StateDevice().api_ = api;
  StateDevice().core_ = core;
  StateDevice().info_ = std::move(info);
}

rhi::IImage *IBackendDevice::CommonCreateImage(const ImageDesc &) {
  ReportError(-1, "Independent images are not implemented for this backend.");
  return nullptr;
}

rhi::IBuffer *IBackendDevice::CommonCreateBuffer(const BufferDesc &) {
  ReportError(-1, "Buffers are not implemented for this backend.");
  return nullptr;
}

rhi::IComputeProgram *IBackendDevice::CommonCreateComputeProgram(const ComputeProgramDesc &) {
  ReportError(-1, "Compute programs are not implemented for this backend.");
  return nullptr;
}

rhi::IProgram *IBackendDevice::CommonCreateProgram(const ProgramDesc &) {
  ReportError(-1, "Graphics programs are not implemented for this backend.");
  return nullptr;
}

rhi::IShader *IBackendDevice::CommonCreateShader(const ShaderDesc &desc) {
  if (desc.filename.empty() || desc.entry_point.empty() || desc.stage < SHADER_STAGE_VERTEX ||
      desc.stage > SHADER_STAGE_CALLABLE)
    return nullptr;
  return IBackendShader::Create(desc, StateDevice().api_);
}

ISampler *IBackendDevice::CommonCreateSampler(const SamplerDesc &) {
  ReportError(-1, "Samplers are not implemented for this backend.");
  return nullptr;
}

IBindingGroup *IBackendDevice::CommonCreateBindingGroup(const BindingLayoutDesc &) {
  ReportError(-1, "Binding groups are not implemented for this backend.");
  return nullptr;
}

void IBackendDevice::DestroyDevice() {
  ShutdownQueues();
}

std::string IBackendDevice::CommonName() const {
  return StateDevice().info_.name;
}

bool IBackendDevice::CommonSupportRayTracingPipeline() const {
  return StateDevice().info_.ray_tracing_pipeline_feature;
}

bool IBackendDevice::CommonSupportRayQuery() const {
  return StateDevice().info_.ray_query_feature;
}

bool IBackendDevice::CommonSupportGeometryShader() const {
  return StateDevice().info_.geometry_shader_feature;
}

rhi::IWindow *IBackendDevice::CommonCreateGLFWWindow(const WindowDesc &) {
  ReportError(-1, "Window creation is unavailable.");
  return nullptr;
}

rhi::IQueue *IBackendDevice::CommonGetQueue(QueueType type) {
  if (type != QUEUE_TYPE_MAIN && type != QUEUE_TYPE_ASSIST)
    return nullptr;
  return ObjectPtr<IBackendQueue>::Retain(StateDevice().queues_[type].Get()).Detach();
}

const PhysicalDevice &IBackendDevice::GetPhysicalDevice() const noexcept {
  return StateDevice().info_;
}

IBackendCore *IBackendDevice::OwnerCore() const {
  return StateDevice().core_;
}

void IBackendDevice::StartCollector() {
  StateDevice().collector_ = std::thread([this] { Collect(); });
}

bool IBackendDevice::CommonWaitEvents(const std::vector<Event> &events, bool wait_all) const {
  if (events.empty())
    return wait_all;
  std::vector<Event> unique;
  unique.reserve(events.size());
  for (const auto &event : events) {
    auto timeline = rhi::QueryInterfacePtr<IBackendTimeline>(event.timeline).Get();
    if (!timeline || timeline->Owner() != this) {
      ReportError(-1, "Host waits require timelines from this device.");
      return false;
    }
    auto found = std::find_if(unique.begin(), unique.end(),
                              [&](const Event &other) { return other.timeline == event.timeline; });
    if (found == unique.end())
      unique.push_back(event);
    else
      found->value = wait_all ? std::max(found->value, event.value) : std::min(found->value, event.value);
  }
  return WaitNativeEvents(unique, wait_all);
}

void IBackendDevice::Collect() {
  std::vector<Event> next;
  next.reserve(StateDevice().queues_.size());
  for (;;) {
    next.clear();
    bool empty = true;
    for (const auto &queue : StateDevice().queues_) {
      if (!queue)
        continue;
      uint64_t completed = 0;
      const bool timeline_valid = queue->GetTimeline()->GetCompletedValue(completed);
      if (timeline_valid && completed != UINT64_MAX)
        next.push_back({queue->GetTimeline(), completed + 1});
      if (!queue->CollectCompleted(completed, timeline_valid))
        empty = false;
    }
    if (StateDevice().stopping_ && empty)
      return;
    WaitForProgress(next);
  }
}

void IBackendDevice::ShutdownQueues() {
  if (!StateDevice().collector_.joinable())
    return;
  StateDevice().stopping_ = true;
  InterruptCollector();
  StateDevice().collector_.join();
}

}  // namespace grassland::rhi::backend

LONGMARCH_RHI_UUID(grassland::rhi::backend::IBackendDevice, "6d6cc71b-9a19-4b0d-ac68-84d86e402f5d")

namespace grassland::rhi::backend {
DeviceState::DeviceState() = default;
DeviceState::~DeviceState() = default;
}  // namespace grassland::rhi::backend
