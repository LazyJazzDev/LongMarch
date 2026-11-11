#pragma once
#include "grassland/rhi/backend/backend_window_state.h"
#include "grassland/rhi/backend/metal/metal_util.h"
#include "grassland/rhi/backend/metal/metal_window.h"

namespace grassland::rhi::backend::metal {
class MetalWindowImpl final : public Object<IMetalWindow> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IWindow, IMetalWindow, IBackendWindow, rhi::IWindow)
  ~MetalWindowImpl() override;
  static IMetalWindow *Create(IMetalDevice *device, const WindowDesc &desc);
  ObjectPtr<IBackendImage> AcquireNativeFrame() override;
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

 public:
  bool EnableImGui(bool enabled = true, const char *font_file_path = nullptr, float font_size = 13.0f) override {
    return IBackendWindow::CommonEnableImGui(enabled, font_file_path, font_size);
  }

  bool UpdateImGui(const std::function<void()> &callback) override {
    return IBackendWindow::CommonUpdateImGui(callback);
  }

  EventManager<void(int, int)> &ResizeEvent() override {
    return IBackendWindow::CommonResizeEvent();
  }

  EventManager<void(int, int)> &FramebufferResizeEvent() override {
    return IBackendWindow::CommonFramebufferResizeEvent();
  }

  EventManager<void(bool)> &CursorEnterEvent() override {
    return IBackendWindow::CommonCursorEnterEvent();
  }

  EventManager<void(bool)> &FocusEvent() override {
    return IBackendWindow::CommonFocusEvent();
  }

  EventManager<void(double, double)> &MouseMoveEvent() override {
    return IBackendWindow::CommonMouseMoveEvent();
  }

  EventManager<void(int, int, int, double, double)> &MouseButtonEvent() override {
    return IBackendWindow::CommonMouseButtonEvent();
  }

  EventManager<void(double, double)> &ScrollEvent() override {
    return IBackendWindow::CommonScrollEvent();
  }

  EventManager<void(int, int, int, int)> &KeyEvent() override {
    return IBackendWindow::CommonKeyEvent();
  }

  EventManager<void(uint32_t)> &CharEvent() override {
    return IBackendWindow::CommonCharEvent();
  }

  EventManager<void(int, const char **)> &DropEvent() override {
    return IBackendWindow::CommonDropEvent();
  }

  rhi::IImage *GetNextImage() override {
    return IBackendWindow::CommonGetNextImage();
  }

  bool IsSwapChainInvalid() const override {
    return IBackendWindow::CommonIsSwapChainInvalid();
  }

  bool RebuildSwapChain() override {
    return IBackendWindow::CommonRebuildSwapChain();
  }

  bool UpdateNextImage() override {
    return IBackendWindow::CommonUpdateNextImage();
  }

  bool SetPresentMode(PresentMode mode) override {
    return IBackendWindow::CommonSetPresentMode(mode);
  }

  bool SetColorSpace(ColorSpace color_space) override {
    return IBackendWindow::CommonSetColorSpace(color_space);
  }

  ColorSpace GetColorSpace() const override {
    return IBackendWindow::CommonGetColorSpace();
  }

  bool GetHDRReferenceWhiteNits(float &nits) const override {
    return IBackendWindow::CommonGetHDRReferenceWhiteNits(nits);
  }

  float GetHDRReferenceWhiteScale() const override {
    return IBackendWindow::CommonGetHDRReferenceWhiteScale();
  }

  float GetHDRHeadroom() const override {
    return IBackendWindow::CommonGetHDRHeadroom();
  }

  GLFWwindow *GLFWWindow() const override {
    return IBackendWindow::CommonGLFWWindow();
  }

  bool ShouldClose() const override {
    return IBackendWindow::CommonShouldClose();
  }

  bool IsKeyDown(int key) const override {
    return IBackendWindow::CommonIsKeyDown(key);
  }

  bool SetTitle(const std::string &title) override {
    return IBackendWindow::CommonSetTitle(title);
  }
};
}  // namespace grassland::rhi::backend::metal
