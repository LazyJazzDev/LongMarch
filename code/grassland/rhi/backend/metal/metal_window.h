#pragma once
#include <QuartzCore/QuartzCore.hpp>

#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::metal::IMetalWindow,
                        grassland::rhi::backend::IBackendWindow,
                        "e074a51b-c30d-43f4-8b6c-2d8fd1540424")

namespace grassland::rhi::backend::metal {

class IMetalWindow : public IBackendWindow {
 public:
  virtual IMetalQueue *OwnerQueue() const = 0;
  virtual bool RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image) = 0;
  virtual bool RecordImGui(rhi::ICommandContext *commands,
                           rhi::IImage *image,
                           ColorSpace color_space,
                           float white_scale) = 0;
  virtual bool RecordPresentImage(ICommandContext *commands, IImage *image, ColorSpace color_space) = 0;
  virtual const void *DeviceIdentity() const = 0;
  virtual bool UpdateNextImage() = 0;
  virtual void ReleaseNextImage() = 0;
  virtual ObjectPtr<IMetalImage> AcquireNativeFrame() = 0;
  virtual bool RebuildNativeSwapChain(PresentMode mode) = 0;
  static rhi::IWindow *Create(IMetalDevice *device, const WindowDesc &desc);

  virtual bool IsFrameValid(uint64_t generation) const = 0;
};

}  // namespace grassland::rhi::backend::metal
