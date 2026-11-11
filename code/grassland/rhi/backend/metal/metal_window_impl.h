#pragma once
#include "grassland/rhi/backend/backend_window_state.h"
#include "grassland/rhi/backend/metal/metal_window.h"

namespace grassland::rhi::backend::metal {
class MetalWindowImpl final : public Object<IMetalWindow> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IWindow, IMetalWindow, Window, rhi::IWindow)
  ~MetalWindowImpl() override;
  static IMetalWindow *Create(IMetalDevice *device, const WindowDesc &desc);
  ObjectPtr<Image> AcquireNativeFrame() override;
  bool RebuildNativeSwapChain(PresentMode mode) override;
  PresentMode GetPresentMode() const override;
  std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const override;
  bool IsFrameValid(uint64_t generation) const;

 private:
  MetalWindowImpl(IMetalDevice *device);
  IMetalDevice *device_;
  NS::SharedPtr<CA::MetalLayer> layer_;
  ImageDesc desc_;
  uint64_t generation_{1};
  bool valid_{};
  PresentMode mode_{PRESENT_MODE_VSYNC};

 private:
  WindowState state_Window_;

  WindowState &StateWindow() override {
    return state_Window_;
  }

  const WindowState &StateWindow() const override {
    return state_Window_;
  }
};
}  // namespace grassland::rhi::backend::metal
