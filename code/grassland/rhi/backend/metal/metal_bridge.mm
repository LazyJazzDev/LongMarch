#include "grassland/rhi/metal_bridge.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_image.h"
#include "grassland/rhi/backend/metal/metal_queue.h"
#include "grassland/rhi/rhi_submission_token.h"

namespace grassland::rhi {
void *GetNativeMetalDevice(IDevice *device) {
  auto native = QueryInterfacePtr<backend::metal::IMetalDevice>(device);
  return native ? native->NativeDevice() : nullptr;
}

bool PresentMetalDrawable(IDevice *device, IImage *source, ISubmissionToken *rendered, void *drawable) {
  @autoreleasepool {
    auto native = QueryInterfacePtr<backend::metal::IMetalDevice>(device);
    auto image = QueryInterfacePtr<backend::metal::IMetalImage>(source);
    if (!native || !image || !source->IsValid() || !rendered || !drawable || image->DeviceIdentity() != native.Get())
      return false;
    id<CAMetalDrawable> target = (__bridge id<CAMetalDrawable>)drawable;
    id<MTLTexture> input = (__bridge id<MTLTexture>)image->NativeImage();
    id<MTLTexture> output = target.texture;
    if (!input || !output || input.device != output.device || input.width != output.width ||
        input.height != output.height || input.pixelFormat != output.pixelFormat || output.framebufferOnly ||
        input.sampleCount != 1 || output.sampleCount != 1)
      return false;
    const auto event = rendered->GetEvent();
    auto timeline = QueryInterfacePtr<backend::metal::IMetalTimeline>(event.timeline);
    if (!timeline || timeline->Owner() != native.Get() || !timeline->NativeEvent() || !event.value)
      return false;
    if (rendered->GetStatus() == SUBMISSION_STATUS_FAILED || rendered->GetStatus() == SUBMISSION_STATUS_CANCELED)
      return false;
    auto queue = native->GetNativeQueue(QUEUE_TYPE_MAIN);
    NS::SharedPtr<MTL::CommandBuffer> commands;
    {
      std::lock_guard lock(queue->QueueMutex());
      commands = queue->CreateCommandBuffer();
      if (!commands)
        return false;
      id<MTLCommandBuffer> buffer = (__bridge id<MTLCommandBuffer>)commands.get();
      commands->encodeWait(timeline->NativeEvent(), event.value);
      id<MTLBlitCommandEncoder> blit = [buffer blitCommandEncoder];
      if (!blit)
        return false;
      [blit copyFromTexture:input
                sourceSlice:0
                sourceLevel:0
               sourceOrigin:MTLOriginMake(0, 0, 0)
                 sourceSize:MTLSizeMake(input.width, input.height, 1)
                  toTexture:output
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
      [blit endEncoding];
      [buffer presentDrawable:target];
      [buffer commit];
    }
    commands->waitUntilCompleted();
    return commands->status() == MTL::CommandBufferStatusCompleted && rendered->WaitHost();
  }
}
}  // namespace grassland::rhi
