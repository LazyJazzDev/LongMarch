#pragma once
#include <QuartzCore/QuartzCore.hpp>

#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/metal/metal_device.h"

namespace grassland::rhi::backend::metal {

class MetalWindow final : public Window {
 public:
  explicit MetalWindow(MetalDevice *device);
  ~MetalWindow() override;
  bool Initialize(const WindowDesc &desc);
  ObjectPtr<Image> AcquireNativeFrame() override;
  bool RebuildNativeSwapChain(PresentMode mode) override;
  PresentMode GetPresentMode() const override;
  std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const override;
  bool IsFrameValid(uint64_t generation) const;
  const void *DeviceIdentity() const;
  void DidPresent();

 private:
  MetalDevice *device_;
  NS::SharedPtr<CA::MetalLayer> layer_;
  ImageDesc desc_;
  uint64_t generation_{1};
  bool acquired_{};
  bool valid_{};
  PresentMode mode_{PRESENT_MODE_VSYNC};
};

}  // namespace grassland::rhi::backend::metal
