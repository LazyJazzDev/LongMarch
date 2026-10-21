#include "grassland/rhi/backend/metal/metal_device.h"

#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/metal/metal_allocated_image.h"
#include "grassland/rhi/backend/metal/metal_binding_group.h"
#include "grassland/rhi/backend/metal/metal_command_context.h"
#include "grassland/rhi/backend/metal/metal_program.h"
#include "grassland/rhi/backend/metal/metal_queue.h"
#include "grassland/rhi/backend/metal/metal_sampler.h"
#include "grassland/rhi/backend/metal/metal_window.h"

namespace grassland::rhi::backend::metal {

MetalDevice *MetalDevice::Create(PhysicalDevice info, Core *core, NS::SharedPtr<MTL::Device> device) {
  auto object = std::unique_ptr<MetalDevice>(new MetalDevice(std::move(info), core, std::move(device)));
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  object->queue_ = NS::TransferPtr(object->device_->newCommandQueue());
  if (!object->queue_) {
    ReportError(-1, "Metal command queue creation failed.");
    return nullptr;
  }
  return object.release();
}

MetalDevice::MetalDevice(PhysicalDevice info, Core *core, NS::SharedPtr<MTL::Device> device)
    : Device(std::move(info), core),
      device_(std::move(device)) {
}

MetalDevice::~MetalDevice() {
  ShutdownQueues();
  uploads_.Clear();
}

MTL::Device *MetalDevice::NativeDevice() const noexcept {
  return device_.get();
}

MTL::CommandQueue *MetalDevice::NativeQueue() const {
  return queue_.get();
}

std::mutex &MetalDevice::QueueMutex() {
  return queue_mutex_;
}

bool MetalDevice::WaitIdle() {
  // Caller serializes this marker with commits using QueueMutex().
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto marker = NS::RetainPtr(queue_->commandBuffer());
  if (!marker) {
    ReportError(-1, "Metal idle marker allocation failed.");
    return false;
  }
  marker->commit();
  marker->waitUntilCompleted();
  return marker->status() == MTL::CommandBufferStatusCompleted;
}

rhi::IWindow *MetalDevice::CreateGLFWWindow(const WindowDesc &desc) {
  return MetalWindow::Create(this, desc);
}

rhi::ICommandContext *MetalDevice::CreateCommandContext() {
  return MetalCommandContext::Create(this);
}

rhi::IQueue *MetalDevice::GetQueue(QueueType type) {
  return new MetalQueue(this, type);
}

rhi::IBuffer *MetalDevice::CreateBuffer(const BufferDesc &desc) {
  return MetalAllocatedBuffer::Create(this, desc);
}

rhi::IImage *MetalDevice::CreateImage(const ImageDesc &desc) {
  return MetalAllocatedImage::Create(this, desc);
}

rhi::IProgram *MetalDevice::CreateProgram(const ProgramDesc &desc) {
  return new MetalProgram(this, desc);
}

rhi::IBindingGroup *MetalDevice::CreateBindingGroup(const BindingLayoutDesc &layout) {
  return MetalBindingGroup::Create(this, layout);
}

ISampler *MetalDevice::CreateSampler(const SamplerDesc &desc) {
  return MetalSampler::Create(this, desc);
}

MetalUploadPool &MetalDevice::UploadPool() {
  return uploads_;
}
}  // namespace grassland::rhi::backend::metal
