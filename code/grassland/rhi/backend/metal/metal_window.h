#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalWindow)
#include <QuartzCore/QuartzCore.hpp>

#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {

class IMetalWindow : public IBackendWindow {
 public:
  static IMetalWindow *Create(IMetalDevice *device, const WindowDesc &desc);
  virtual ObjectPtr<IBackendImage> AcquireNativeFrame() = 0;
  virtual bool RebuildNativeSwapChain(PresentMode mode) = 0;
  virtual PresentMode GetPresentMode() const = 0;
  virtual std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const = 0;
  virtual bool IsFrameValid(uint64_t generation) const = 0;
};

;

}  // namespace grassland::rhi::backend::metal
