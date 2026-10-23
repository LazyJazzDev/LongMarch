#include "grassland/rhi/backend/metal/metal_window.h"

#include "grassland/rhi/backend/metal/metal_swap_chain_image.h"
#define GLFW_EXPOSE_NATIVE_COCOA
#import <Cocoa/Cocoa.h>
#include <GLFW/glfw3native.h>
#import <QuartzCore/CAMetalLayer.h>

namespace grassland::rhi::backend::metal {

MetalWindow::MetalWindow(MetalDevice *device) : device_(device) {
}

MetalWindow::~MetalWindow() {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  glfw::UnregisterWindow(GLFWWindow());
  std::lock_guard<std::mutex> lock(device_->QueueMutex());
  device_->WaitIdle();
  valid_ = false;
  ReleaseNextImage();
  if (GLFWWindow())
    [[glfwGetCocoaWindow(GLFWWindow()) contentView] setLayer:nil];
  layer_.reset();
}

MetalWindow *MetalWindow::Create(MetalDevice *device, const WindowDesc &desc) {
  auto object = std::unique_ptr<MetalWindow>(new MetalWindow(device));
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  if (!object->InitializeWindow(device, desc))
    return nullptr;
  object->layer_ = NS::RetainPtr(CA::MetalLayer::layer());
  if (!object->layer_) {
    ReportError(-1, "CAMetalLayer creation failed.");
    return nullptr;
  }
  object->layer_->setDevice(object->device_->NativeDevice());
  object->layer_->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
  // RHI images may also be read back/copied, rather than only render targets.
  object->layer_->setFramebufferOnly(false);
  auto view = [glfwGetCocoaWindow(object->GLFWWindow()) contentView];
  [view setWantsLayer:YES];
  [view setLayer:(CAMetalLayer *)object->layer_.get()];
  auto colorspace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  ((CAMetalLayer *)object->layer_.get()).colorspace = colorspace;
  if (colorspace)
    CGColorSpaceRelease(colorspace);
  if (!object->RebuildSwapChain())
    return nullptr;
  return object.release();
}

bool MetalWindow::RebuildNativeSwapChain(PresentMode requested_mode) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  std::lock_guard<std::mutex> lock(device_->QueueMutex());
  if (!device_->WaitIdle())
    return false;
  auto layer = (CAMetalLayer *)layer_.get();
  layer.displaySyncEnabled = requested_mode != PRESENT_MODE_CONTINUOUS;
  mode_ = layer.displaySyncEnabled ? PRESENT_MODE_VSYNC : PRESENT_MODE_CONTINUOUS;
  valid_ = false;
  ++generation_;
  if (glfwGetWindowAttrib(GLFWWindow(), GLFW_ICONIFIED))
    return true;
  int width, height;
  glfwGetFramebufferSize(GLFWWindow(), &width, &height);
  if (width <= 0 || height <= 0)
    return true;
  layer_->setDrawableSize(CGSizeMake(width, height));
  NSWindow *native = glfwGetCocoaWindow(GLFWWindow());
  ((CAMetalLayer *)layer_.get()).contentsScale = native.backingScaleFactor;
  desc_ = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), IMAGE_FORMAT_BGRA8_UNORM};
  valid_ = true;
  return true;
}

ObjectPtr<Image> MetalWindow::AcquireNativeFrame() {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  int width, height;
  glfwGetFramebufferSize(GLFWWindow(), &width, &height);
  if (glfwGetWindowAttrib(GLFWWindow(), GLFW_ICONIFIED) || width <= 0 || height <= 0)
    return {};
#if !defined(NDEBUG)
  if (!valid_ || desc_.width != width || desc_.height != height) {
    ReportError(-1, "Cannot acquire a Metal drawable in the current window state.");
    return {};
  }
#endif
  auto drawable = NS::RetainPtr(layer_->nextDrawable());
  if (!drawable) {
    ReportError(-1, "CAMetalLayer::nextDrawable failed.");
    return {};
  }
  return ObjectPtr<Image>::Adopt(new MetalSwapChainImage(desc_, this, generation_, drawable.get()));
}

PresentMode MetalWindow::GetPresentMode() const {
  return mode_;
}

std::vector<WindowPresentationOption> MetalWindow::GetSupportedPresentationOptions() const {
  return {{PRESENT_MODE_VSYNC, IMAGE_FORMAT_BGRA8_UNORM}, {PRESENT_MODE_CONTINUOUS, IMAGE_FORMAT_BGRA8_UNORM}};
}

bool MetalWindow::IsFrameValid(uint64_t generation) const {
  return valid_ && generation_ == generation;
}

const void *MetalWindow::DeviceIdentity() const {
  return device_;
}

}  // namespace grassland::rhi::backend::metal
