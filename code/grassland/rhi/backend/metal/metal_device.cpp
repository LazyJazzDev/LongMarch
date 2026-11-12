#include "grassland/rhi/backend/metal/metal_device.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <list>
#include <mutex>
#include <thread>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/backend_shader.h"
#include "grassland/rhi/backend/backend_shader_impl.h"
#include "grassland/rhi/backend/backend_submission_token.h"
#include "grassland/rhi/backend/metal/metal_acceleration_structure.h"
#include "grassland/rhi/backend/metal/metal_allocated_image.h"
#include "grassland/rhi/backend/metal/metal_binding_group.h"
#include "grassland/rhi/backend/metal/metal_binding_group_impl.h"
#include "grassland/rhi/backend/metal/metal_command_context.h"
#include "grassland/rhi/backend/metal/metal_compute_program.h"
#include "grassland/rhi/backend/metal/metal_core.h"
#include "grassland/rhi/backend/metal/metal_device_impl.h"
#include "grassland/rhi/backend/metal/metal_diagnostics.h"
#include "grassland/rhi/backend/metal/metal_program.h"
#include "grassland/rhi/backend/metal/metal_queue.h"
#include "grassland/rhi/backend/metal/metal_sampler.h"
#include "grassland/rhi/backend/metal/metal_window.h"
#include "grassland/rhi/rhi_device.h"
#include "grassland/rhi/rhi_queue.h"

namespace grassland::rhi::backend::metal {

rhi::IDevice *MetalDeviceImpl::Create(PhysicalDevice info, IMetalCore *core, NS::SharedPtr<MTL::Device> device) {
  auto object = std::unique_ptr<MetalDeviceImpl>(new MetalDeviceImpl(std::move(info), core, std::move(device)));
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  object->wait_state_->listener = NS::TransferPtr(MTL::SharedEventListener::alloc()->init());
  if (!object->wait_state_->listener) {
    ReportError(-1, "Create Metal progress listener failed.");
    return nullptr;
  }
  auto descriptor = NS::TransferPtr(MTL::CommandBufferDescriptor::alloc()->init());
  descriptor->setRetainedReferences(true);
  descriptor->setErrorOptions(core->DebugEnabled() ? MTL::CommandBufferErrorOptionEncoderExecutionStatus
                                                   : MTL::CommandBufferErrorOptionNone);
  for (auto type : {QUEUE_TYPE_MAIN, QUEUE_TYPE_ASSIST}) {
    object->queues_[type] = ObjectPtr<IMetalQueue>::Adopt(IMetalQueue::Create(object.get(), type, descriptor.get()));
    if (!object->queues_[type])
      return nullptr;
  }
  object->StartCollector();
  return object.release();
}

MetalDeviceImpl::MetalDeviceImpl(PhysicalDevice info, IMetalCore *core, NS::SharedPtr<MTL::Device> device)
    : device_(std::move(device)),
      temporal_buffers_(this, TEMPORAL_BUFFER_UPLOAD) {
  InitializeDevice(std::move(info), core, BACKEND_API_METAL);
}

MetalDeviceImpl::~MetalDeviceImpl() {
  ShutdownQueues();
  temporal_buffers_.Clear();
  scratch_.Clear();
  for (auto &queue : queues_)
    queue.Reset();

  DestroyDevice();
}

MetalScratchPool &MetalDeviceImpl::ScratchPool() {
  return scratch_;
}

IBottomLevelAccelerationStructure *MetalDeviceImpl::CreateBottomLevelAccelerationStructure(
    const BottomLevelAccelerationStructureDesc &desc) {
  return CreateMetalBottomLevelAccelerationStructure(this, desc);
}

ITopLevelAccelerationStructure *MetalDeviceImpl::CreateTopLevelAccelerationStructure(
    const TopLevelAccelerationStructureDesc &desc) {
  return IMetalTopLevelAccelerationStructure::Create(this, desc);
}

MTL::Device *MetalDeviceImpl::NativeDevice() const noexcept {
  return device_.get();
}

rhi::IWindow *MetalDeviceImpl::CreateGLFWWindow(const WindowDesc &desc) {
  return IMetalWindow::Create(this, desc);
}

rhi::IBuffer *MetalDeviceImpl::CreateBuffer(const BufferDesc &desc) {
  auto buffer = ObjectPtr<IMetalAllocatedBuffer>::Adopt(IMetalAllocatedBuffer::Create(this, desc));
  return rhi::QueryInterfacePtr<IBuffer>(buffer.Get()).Detach();
}

rhi::IImage *MetalDeviceImpl::CreateImage(const ImageDesc &desc) {
  return CreateMetalAllocatedImage(this, desc);
}

IComputeProgram *MetalDeviceImpl::CreateComputeProgram(const ComputeProgramDesc &desc) {
  return IMetalComputeProgram::Create(this, desc);
}

rhi::IProgram *MetalDeviceImpl::CreateProgram(const ProgramDesc &desc) {
  return IMetalProgram::Create(this, desc);
}

rhi::IBindingGroup *MetalDeviceImpl::CreateBindingGroup(const BindingLayoutDesc &layout) {
  return CreateMetalBindingGroup(this, layout);
}

ISampler *MetalDeviceImpl::CreateSampler(const SamplerDesc &desc) {
  return IMetalSampler::Create(this, desc);
}

MetalTemporalBufferPool &MetalDeviceImpl::UploadPool() {
  return temporal_buffers_;
}

MetalTemporalBufferPool &MetalDeviceImpl::DownloadPool() {
  return temporal_buffers_;
}

rhi::IDevice *IMetalDevice::Create(PhysicalDevice info, IMetalCore *core, NS::SharedPtr<MTL::Device> device) {
  return MetalDeviceImpl::Create(info, core, device);
}

IRayTracingProgram *MetalDeviceImpl::CreateRayTracingProgram(const RayTracingProgramDesc &) {
  ReportError(-1, "Ray tracing pipelines are unavailable for this backend.");
  return nullptr;
}

void MetalDeviceImpl::InitializeDevice(PhysicalDevice info, IMetalCore *core, BackendAPI api) {
  api_ = api;
  core_ = core;
  info_ = std::move(info);
}

rhi::IShader *MetalDeviceImpl::CreateShader(const ShaderDesc &desc) {
  if (desc.filename.empty() || desc.entry_point.empty() || desc.stage < SHADER_STAGE_VERTEX ||
      desc.stage > SHADER_STAGE_CALLABLE)
    return nullptr;
  return backend::CreateShader(desc, api_);
}

void MetalDeviceImpl::DestroyDevice() {
  ShutdownQueues();
}

std::string MetalDeviceImpl::Name() const {
  return info_.name;
}

bool MetalDeviceImpl::SupportRayTracingPipeline() const {
  return info_.ray_tracing_pipeline_feature;
}

bool MetalDeviceImpl::SupportRayQuery() const {
  return info_.ray_query_feature;
}

bool MetalDeviceImpl::SupportGeometryShader() const {
  return info_.geometry_shader_feature;
}

IMetalQueue *MetalDeviceImpl::GetNativeQueue(QueueType type) const {
  if (type != QUEUE_TYPE_MAIN && type != QUEUE_TYPE_ASSIST)
    return nullptr;
  return queues_[type].Get();
}

rhi::IQueue *MetalDeviceImpl::GetQueue(QueueType type) {
  if (type != QUEUE_TYPE_MAIN && type != QUEUE_TYPE_ASSIST)
    return nullptr;
  return rhi::QueryInterfacePtr<IQueue>(queues_[type].Get()).Detach();
}

const PhysicalDevice &MetalDeviceImpl::GetPhysicalDevice() const noexcept {
  return info_;
}

IMetalCore *MetalDeviceImpl::OwnerCore() const {
  return core_;
}

void MetalDeviceImpl::StartCollector() {
  collector_ = std::thread([this] { Collect(); });
}

bool MetalDeviceImpl::WaitEvents(const std::vector<Event> &events, bool wait_all) const {
  if (events.empty())
    return wait_all;
  std::vector<Event> unique;
  unique.reserve(events.size());
  for (const auto &event : events) {
    auto timeline = rhi::QueryInterfacePtr<IMetalTimeline>(event.timeline).Get();
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

void MetalDeviceImpl::Collect() {
  std::vector<Event> next;
  next.reserve(queues_.size());
  for (;;) {
    next.clear();
    bool empty = true;
    for (const auto &queue : queues_) {
      if (!queue)
        continue;
      uint64_t completed = 0;
      auto timeline = rhi::QueryInterfacePtr<ITimeline>(queue->GetTimeline());
      const bool timeline_valid = timeline->GetCompletedValue(completed);
      if (timeline_valid && completed != UINT64_MAX)
        next.push_back({timeline.Get(), completed + 1});
      if (!queue->CollectCompleted(completed, timeline_valid))
        empty = false;
    }
    if (stopping_ && empty)
      return;
    WaitForProgress(next);
  }
}

void MetalDeviceImpl::ShutdownQueues() {
  if (!collector_.joinable())
    return;
  stopping_ = true;
  InterruptCollector();
  collector_.join();
}
}  // namespace grassland::rhi::backend::metal
