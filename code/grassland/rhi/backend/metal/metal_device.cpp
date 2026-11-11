#include "grassland/rhi/backend/metal/metal_device.h"

#include "grassland/rhi/backend/metal/metal_acceleration_structure.h"
#include "grassland/rhi/backend/metal/metal_acceleration_structure_impl.h"
#include "grassland/rhi/backend/metal/metal_allocated_image.h"
#include "grassland/rhi/backend/metal/metal_allocated_image_impl.h"
#include "grassland/rhi/backend/metal/metal_binding_group.h"
#include "grassland/rhi/backend/metal/metal_command_context.h"
#include "grassland/rhi/backend/metal/metal_compute_program.h"
#include "grassland/rhi/backend/metal/metal_compute_program_impl.h"
#include "grassland/rhi/backend/metal/metal_core.h"
#include "grassland/rhi/backend/metal/metal_device_impl.h"
#include "grassland/rhi/backend/metal/metal_diagnostics.h"
#include "grassland/rhi/backend/metal/metal_program.h"
#include "grassland/rhi/backend/metal/metal_program_impl.h"
#include "grassland/rhi/backend/metal/metal_queue.h"
#include "grassland/rhi/backend/metal/metal_queue_impl.h"
#include "grassland/rhi/backend/metal/metal_sampler.h"
#include "grassland/rhi/backend/metal/metal_sampler_impl.h"
#include "grassland/rhi/backend/metal/metal_window.h"
#include "grassland/rhi/backend/metal/metal_window_impl.h"

namespace grassland::rhi::backend::metal {

IMetalDevice *MetalDeviceImpl::Create(PhysicalDevice info, Core *core, NS::SharedPtr<MTL::Device> device) {
  auto object = std::unique_ptr<MetalDeviceImpl>(new MetalDeviceImpl(std::move(info), core, std::move(device)));
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  object->wait_state_->listener = NS::TransferPtr(MTL::SharedEventListener::alloc()->init());
  if (!object->wait_state_->listener) {
    ReportError(-1, "Create Metal progress listener failed.");
    return nullptr;
  }
  auto descriptor = NS::TransferPtr(MTL::CommandBufferDescriptor::alloc()->init());
  descriptor->setRetainedReferences(true);
  descriptor->setErrorOptions(rhi::QueryInterfacePtr<IMetalCore>(core).Get()->DebugEnabled()
                                  ? MTL::CommandBufferErrorOptionEncoderExecutionStatus
                                  : MTL::CommandBufferErrorOptionNone);
  for (auto type : {QUEUE_TYPE_MAIN, QUEUE_TYPE_ASSIST}) {
    object->StateDevice().queues_[type] =
        ObjectPtr<Queue>::Adopt(IMetalQueue::Create(object.get(), type, descriptor.get()));
    if (!object->StateDevice().queues_[type])
      return nullptr;
  }
  object->StartCollector();
  return object.release();
}

MetalDeviceImpl::MetalDeviceImpl(PhysicalDevice info, Core *core, NS::SharedPtr<MTL::Device> device)
    : device_(std::move(device)),
      temporal_buffers_(this, TEMPORAL_BUFFER_UPLOAD) {
  InitializeDevice(std::move(info), core, BACKEND_API_METAL);
}

MetalDeviceImpl::~MetalDeviceImpl() {
  ShutdownQueues();
  temporal_buffers_.Clear();
  scratch_.Clear();
  for (auto &queue : StateDevice().queues_)
    queue.Reset();

  DestroyDevice();
}

MetalScratchPool &MetalDeviceImpl::ScratchPool() {
  return scratch_;
}

IBottomLevelAccelerationStructure *MetalDeviceImpl::CreateBottomLevelAccelerationStructure(
    const BottomLevelAccelerationStructureDesc &desc) {
  return IMetalBottomLevelAccelerationStructure::Create(this, desc);
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
  return IMetalAllocatedBuffer::Create(this, desc);
}

rhi::IImage *MetalDeviceImpl::CreateImage(const ImageDesc &desc) {
  return IMetalAllocatedImage::Create(this, desc);
}

IComputeProgram *MetalDeviceImpl::CreateComputeProgram(const ComputeProgramDesc &desc) {
  return IMetalComputeProgram::Create(this, desc);
}

rhi::IProgram *MetalDeviceImpl::CreateProgram(const ProgramDesc &desc) {
  return IMetalProgram::Create(this, desc);
}

rhi::IBindingGroup *MetalDeviceImpl::CreateBindingGroup(const BindingLayoutDesc &layout) {
  return MetalBindingGroup::Create(this, layout);
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
}  // namespace grassland::rhi::backend::metal

LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::IMetalDevice, "0edf854d-4ccc-4b02-8c0c-030ded3b67dc")

namespace grassland::rhi::backend::metal {
IMetalDevice *IMetalDevice::Create(PhysicalDevice info, Core *core, NS::SharedPtr<MTL::Device> device) {
  return MetalDeviceImpl::Create(info, core, device);
}
}  // namespace grassland::rhi::backend::metal
