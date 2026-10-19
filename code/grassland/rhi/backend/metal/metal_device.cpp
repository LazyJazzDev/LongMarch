#include "grassland/rhi/backend/metal/metal_device.h"

#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/metal/metal_command_context.h"
#include "grassland/rhi/backend/metal/metal_queue.h"
#include "grassland/rhi/backend/metal/metal_window.h"

namespace grassland::rhi::backend::metal {

MetalDevice::MetalDevice(PhysicalDevice info, Core *core, NS::SharedPtr<MTL::Device> device)
    : Device(std::move(info), core),
      device_(std::move(device)) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  queue_ = NS::TransferPtr(device_->newCommandQueue());
  if (!queue_)
    ReportError(-1, "Metal command queue creation failed.");
}

MetalDevice::~MetalDevice() {
  ShutdownQueues();
}

MTL::Device *MetalDevice::NativeDevice() const noexcept {
  return device_.get();
}

bool MetalDevice::IsInitialized() const {
  return bool(queue_);
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
  try {
    auto window = std::make_unique<MetalWindow>(this);
    if (!window->Initialize(desc))
      return nullptr;
    return window.release();
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Window creation ran out of memory.");
    return nullptr;
  }
}

rhi::ICommandContext *MetalDevice::CreateCommandContext() {
  try {
    auto context = std::make_unique<MetalCommandContext>(this);
    if (!context->Initialize())
      return nullptr;
    return context.release();
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Command context allocation failed.");
    return nullptr;
  }
}

rhi::IQueue *MetalDevice::GetQueue(QueueType type) {
  try {
    return new MetalQueue(this, type);
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Queue creation ran out of memory.");
    return nullptr;
  }
}

}  // namespace grassland::rhi::backend::metal
