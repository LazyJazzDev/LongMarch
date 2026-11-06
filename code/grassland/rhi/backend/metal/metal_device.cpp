#include "grassland/rhi/backend/metal/metal_device.h"

#include "grassland/rhi/backend/metal/metal_acceleration_structure.h"
#include "grassland/rhi/backend/metal/metal_allocated_image.h"
#include "grassland/rhi/backend/metal/metal_binding_group.h"
#include "grassland/rhi/backend/metal/metal_command_context.h"
#include "grassland/rhi/backend/metal/metal_compute_program.h"
#include "grassland/rhi/backend/metal/metal_core.h"
#include "grassland/rhi/backend/metal/metal_diagnostics.h"
#include "grassland/rhi/backend/metal/metal_program.h"
#include "grassland/rhi/backend/metal/metal_queue.h"
#include "grassland/rhi/backend/metal/metal_sampler.h"
#include "grassland/rhi/backend/metal/metal_window.h"

namespace grassland::rhi::backend::metal {

MetalDevice *MetalDevice::Create(PhysicalDevice info, Core *core, NS::SharedPtr<MTL::Device> device) {
  auto object = std::unique_ptr<MetalDevice>(new MetalDevice(std::move(info), core, std::move(device)));
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  object->command_buffer_descriptor_ = NS::TransferPtr(MTL::CommandBufferDescriptor::alloc()->init());
  object->command_buffer_descriptor_->setRetainedReferences(true);
  object->command_buffer_descriptor_->setErrorOptions(static_cast<MetalCore *>(core)->DebugEnabled()
                                                          ? MTL::CommandBufferErrorOptionEncoderExecutionStatus
                                                          : MTL::CommandBufferErrorOptionNone);
  for (auto type : {QUEUE_TYPE_GRAPHICS, QUEUE_TYPE_TRANSFER}) {
    auto queue = ObjectPtr<MetalQueue>::Adopt(MetalQueue::Create(object.get(), type));
    if (!queue)
      return nullptr;
    object->InitializeQueue(std::move(queue));
  }
  return object.release();
}

MetalDevice::MetalDevice(PhysicalDevice info, Core *core, NS::SharedPtr<MTL::Device> device)
    : Device(std::move(info), core, BACKEND_API_METAL),
      device_(std::move(device)),
      temporal_buffers_(this, TEMPORAL_BUFFER_UPLOAD) {
}

MetalDevice::~MetalDevice() {
  ShutdownQueues();
  temporal_buffers_.Clear();
  scratch_.Clear();
  ReleaseQueues();
}

MetalScratchPool &MetalDevice::ScratchPool() {
  return scratch_;
}

IBottomLevelAccelerationStructure *MetalDevice::CreateBottomLevelAccelerationStructure(
    const BottomLevelAccelerationStructureDesc &desc) {
  return MetalBottomLevelAccelerationStructure::Create(this, desc);
}

ITopLevelAccelerationStructure *MetalDevice::CreateTopLevelAccelerationStructure(
    const TopLevelAccelerationStructureDesc &desc) {
  return MetalTopLevelAccelerationStructure::Create(this, desc);
}

MTL::Device *MetalDevice::NativeDevice() const noexcept {
  return device_.get();
}

MTL::CommandQueue *MetalDevice::NativeQueue(QueueType type) const {
  return static_cast<MetalQueue *>(QueueState(type))->Native();
}

std::mutex &MetalDevice::QueueMutex(QueueType type) {
  return QueueState(type)->Mutex();
}

const MTL::CommandBufferDescriptor *MetalDevice::CommandBufferDescriptor() const {
  return command_buffer_descriptor_.get();
}

NS::SharedPtr<MTL::CommandBuffer> MetalDevice::CreateCommandBuffer(QueueType type) {
  return static_cast<MetalQueue *>(QueueState(type))->CreateCommandBuffer();
}

bool MetalDevice::WaitIdle() {
  // Caller serializes this marker with commits using QueueMutex().
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
                     static_cast<MetalCore *>(OwnerCore())->DebugEnabled());
    return false;
  }
  return true;
}

rhi::IWindow *MetalDevice::CreateGLFWWindow(const WindowDesc &desc) {
  return MetalWindow::Create(this, desc);
}

rhi::IBuffer *MetalDevice::CreateBuffer(const BufferDesc &desc) {
  return MetalAllocatedBuffer::Create(this, desc);
}

rhi::IImage *MetalDevice::CreateImage(const ImageDesc &desc) {
  return MetalAllocatedImage::Create(this, desc);
}

IComputeProgram *MetalDevice::CreateComputeProgram(const ComputeProgramDesc &desc) {
  return MetalComputeProgram::Create(this, desc);
}

rhi::IProgram *MetalDevice::CreateProgram(const ProgramDesc &desc) {
  return MetalProgram::Create(this, desc);
}

rhi::IBindingGroup *MetalDevice::CreateBindingGroup(const BindingLayoutDesc &layout) {
  return MetalBindingGroup::Create(this, layout);
}

ISampler *MetalDevice::CreateSampler(const SamplerDesc &desc) {
  return MetalSampler::Create(this, desc);
}

MetalTemporalBufferPool &MetalDevice::UploadPool() {
  return temporal_buffers_;
}

MetalTemporalBufferPool &MetalDevice::DownloadPool() {
  return temporal_buffers_;
}
}  // namespace grassland::rhi::backend::metal

namespace grassland::rhi::backend::metal {

MTL::SharedEvent *MetalDevice::Timeline(QueueType type) const {
  return static_cast<MetalQueue *>(QueueState(type))->Timeline();
}

uint64_t MetalDevice::NextSerial(QueueType type) {
  return QueueState(type)->NextSerial();
}

bool MetalDevice::WaitTimelines(const std::vector<Event> &points, bool wait_all) const {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  std::unique_lock<std::mutex> lock(progress_mutex_);
  bool valid = true;
  progress_changed_.wait(lock, [&] {
    bool all = true, any = false;
    for (auto p : points) {
      uint64_t value{};
      valid &= p.timeline->GetCompletedValue(value);
      bool done = value >= p.value;
      all &= done;
      any |= done;
    }
    return !valid || (wait_all ? all : any);
  });
  return valid;
}

void MetalDevice::WakeCollector() {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  std::vector<NS::SharedPtr<MTL::CommandBuffer>> markers;
  for (auto type : {QUEUE_TYPE_GRAPHICS, QUEUE_TYPE_TRANSFER}) {
    std::lock_guard<std::mutex> lock(QueueMutex(type));
    auto marker = CreateCommandBuffer(type);
    if (!marker)
      continue;
    marker->encodeSignalEvent(Timeline(type), NextSerial(type));
    TrackCompletion(marker.get(), QueueState(type)->TimelineState());
    marker->commit();
    markers.push_back(std::move(marker));
  }
  for (const auto &marker : markers)
    marker->waitUntilCompleted();
}

}  // namespace grassland::rhi::backend::metal
