#include "grassland/rhi/backend/metal/metal_window.h"

#include "grassland/rhi/backend/metal/metal_swap_chain_image.h"
#include "grassland/rhi/backend/metal/metal_util.h"
#define GLFW_EXPOSE_NATIVE_COCOA
#import <Cocoa/Cocoa.h>
#include <GLFW/glfw3native.h>
#import <QuartzCore/CAMetalLayer.h>

namespace grassland::rhi::backend::metal {
namespace {
ImageFormat PresentationFormat(ColorSpace space) {
  switch (space) {
    case COLOR_SPACE_SRGB_NONLINEAR:
      return IMAGE_FORMAT_BGRA8_UNORM;
    case COLOR_SPACE_EXTENDED_SRGB_LINEAR:
      return IMAGE_FORMAT_RGBA16_FLOAT;
    case COLOR_SPACE_HDR10_ST2084:
      return IMAGE_FORMAT_RGB10A2_UNORM;
    default:
      return IMAGE_FORMAT_UNKNOWN;
  }
}

CGColorSpaceRef CreatePresentationColorSpace(ColorSpace space) {
  switch (space) {
    case COLOR_SPACE_SRGB_NONLINEAR:
      return CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    case COLOR_SPACE_EXTENDED_SRGB_LINEAR:
      return CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearSRGB);
    case COLOR_SPACE_HDR10_ST2084:
      return CGColorSpaceCreateWithName(kCGColorSpaceITUR_2100_PQ);
    default:
      return nullptr;
  }
}
}  // namespace

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
  if (!object->InitializeWindow(desc))
    return nullptr;
  object->layer_ = NS::RetainPtr(CA::MetalLayer::layer());
  if (!object->layer_) {
    ReportError(-1, "CAMetalLayer creation failed.");
    return nullptr;
  }
  object->layer_->setDevice(object->device_->NativeDevice());
  // RHI images may also be read back/copied, rather than only render targets.
  object->layer_->setFramebufferOnly(false);
  auto view = [glfwGetCocoaWindow(object->GLFWWindow()) contentView];
  [view setWantsLayer:YES];
  [view setLayer:(CAMetalLayer *)object->layer_.get()];
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
  const auto format = PresentationFormat(RequestedColorSpace());
  auto color_space = CreatePresentationColorSpace(RequestedColorSpace());
  if (!color_space || format == IMAGE_FORMAT_UNKNOWN) {
    if (color_space)
      CGColorSpaceRelease(color_space);
    ReportError(-1, "Metal presentation color space is unavailable.");
    return false;
  }
  layer.pixelFormat = static_cast<MTLPixelFormat>(NativeImageFormat(format));
  layer.colorspace = color_space;
  layer.wantsExtendedDynamicRangeContent = RequestedColorSpace() != COLOR_SPACE_SRGB_NONLINEAR;
  CGColorSpaceRelease(color_space);
  layer_->setDrawableSize(CGSizeMake(width, height));
  NSWindow *native = glfwGetCocoaWindow(GLFWWindow());
  ((CAMetalLayer *)layer_.get()).contentsScale = native.backingScaleFactor;
  desc_ = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), format};
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
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  std::vector<WindowPresentationOption> options;
  // EDR content is accepted even on SDR displays; current display headroom controls
  // visible luminance, not the layer's supported input encodings.
  for (auto space : {COLOR_SPACE_SRGB_NONLINEAR, COLOR_SPACE_EXTENDED_SRGB_LINEAR, COLOR_SPACE_HDR10_ST2084}) {
    auto color_space = CreatePresentationColorSpace(space);
    if (!color_space)
      continue;
    CGColorSpaceRelease(color_space);
    for (auto mode : {PRESENT_MODE_VSYNC, PRESENT_MODE_CONTINUOUS})
      options.push_back({mode, PresentationFormat(space), space});
  }
  return options;
}

bool MetalWindow::IsFrameValid(uint64_t generation) const {
  return valid_ && generation_ == generation;
}

const void *MetalWindow::DeviceIdentity() const {
  return device_;
}

}  // namespace grassland::rhi::backend::metal
