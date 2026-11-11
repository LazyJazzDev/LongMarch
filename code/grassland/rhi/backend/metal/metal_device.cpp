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
  object->wait_state_->listener = NS::TransferPtr(MTL::SharedEventListener::alloc()->init());
  if (!object->wait_state_->listener) {
    ReportError(-1, "Create Metal progress listener failed.");
    return nullptr;
  }
  auto descriptor = NS::TransferPtr(MTL::CommandBufferDescriptor::alloc()->init());
  descriptor->setRetainedReferences(true);
  descriptor->setErrorOptions(rhi::QueryInterfacePtr<MetalCore>(core).Get()->DebugEnabled()
                                  ? MTL::CommandBufferErrorOptionEncoderExecutionStatus
                                  : MTL::CommandBufferErrorOptionNone);
  for (auto type : {QUEUE_TYPE_MAIN, QUEUE_TYPE_ASSIST}) {
    object->queues_[type] = ObjectPtr<Queue>::Adopt(MetalQueue::Create(object.get(), type, descriptor.get()));
    if (!object->queues_[type])
      return nullptr;
  }
  object->StartCollector();
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
  for (auto &queue : queues_)
    queue.Reset();
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
