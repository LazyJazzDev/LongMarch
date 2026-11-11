#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::metal {
class MetalWindow;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::MetalWindow)
#include <QuartzCore/QuartzCore.hpp>

#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {

class MetalWindow final : public Window {
 public:
  LONGMARCH_RHI_QUERY(MetalWindow, Window)

  ~MetalWindow() override;
  static MetalWindow *Create(MetalDevice *device, const WindowDesc &desc);
  ObjectPtr<Image> AcquireNativeFrame() override;
  bool RebuildNativeSwapChain(PresentMode mode) override;
  PresentMode GetPresentMode() const override;
  std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const override;
  bool IsFrameValid(uint64_t generation) const;

 private:
  MetalWindow(MetalDevice *device);

  MetalDevice *device_;
  NS::SharedPtr<CA::MetalLayer> layer_;
  ImageDesc desc_;
  uint64_t generation_{1};
  bool valid_{};
  PresentMode mode_{PRESENT_MODE_VSYNC};
};

}  // namespace grassland::rhi::backend::metal
