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
  static rhi::IWindow *Create(IMetalDevice *device, const WindowDesc &desc);

  virtual bool IsFrameValid(uint64_t generation) const = 0;
};

}  // namespace grassland::rhi::backend::metal

namespace grassland::rhi::backend::metal {
class MetalWindowImpl final : public Object<rhi::IWindow, IMetalWindow> {
 public:
  ~MetalWindowImpl() override;
  static rhi::IWindow *Create(IMetalDevice *device, const WindowDesc &desc);
  PresentMode GetPresentMode() const override;
  std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const override;
  bool IsFrameValid(uint64_t generation) const;

 private:
  void ReleaseNextImage();
  ObjectPtr<IMetalImage> AcquireNativeFrame();
  bool RebuildNativeSwapChain(PresentMode mode);
  MetalWindowImpl(IMetalDevice *device);
  IMetalDevice *device_;
  NS::SharedPtr<CA::MetalLayer> layer_;
  ImageDesc desc_;
  uint64_t generation_{1};
  bool valid_{};
  PresentMode mode_{PRESENT_MODE_VSYNC};

 public:
  bool EnableImGui(bool enabled = true, const char *font_file_path = nullptr, float font_size = 13.0f) override;

  ImGuiContext *GetImGuiContext() const override;

  bool UpdateImGui(const std::function<void()> &callback) override;

  const void *DeviceIdentity() const override;

  bool RecordPresentImage(ICommandContext *commands, IImage *image, ColorSpace color_space) override;

  bool RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image) override;

  bool RecordImGui(rhi::ICommandContext *commands,
                   rhi::IImage *image,
                   ColorSpace color_space,
                   float white_scale) override;

  rhi::IImage *GetNextImage() override;

  IMetalQueue *OwnerQueue() const override;

  bool UpdateNextImage() override;

  bool IsSwapChainInvalid() const override;

  bool RebuildSwapChain() override;

  bool SetPresentMode(PresentMode mode) override;

  bool SetColorSpace(ColorSpace color_space) override;

  ColorSpace GetColorSpace() const override;

  float GetHDRReferenceWhiteScale() const override;

  float GetHDRHeadroom() const override;

  bool GetHDRReferenceWhiteNits(float &nits) const override;

 private:
  ColorSpace RequestedColorSpace() const;

  void DestroyWindow();

  bool InitializeWindow(IMetalDevice *device, const WindowDesc &desc);

 public:
  GLFWwindow *GLFWWindow() const override;

  bool ShouldClose() const override;

  bool IsKeyDown(int key) const override;

  bool SetTitle(const std::string &title) override;

  EventManager<void(int, int)> &ResizeEvent() override;

  EventManager<void(int, int)> &FramebufferResizeEvent() override;

  EventManager<void(bool)> &CursorEnterEvent() override;

  EventManager<void(bool)> &FocusEvent() override;

  EventManager<void(double, double)> &MouseMoveEvent() override;

  EventManager<void(int, int, int, double, double)> &MouseButtonEvent() override;

  bool SupportsMagnifyGestures() const override {
#if defined(__APPLE__)
    return magnify_monitor_ != nullptr;
#else
    return false;
#endif
  }

  EventManager<void(const MagnifyGesture &)> &MagnifyEvent() override {
    return magnify_event_;
  }

  EventManager<void(double, double)> &ScrollEvent() override;

  EventManager<void(int, int, int, int)> &KeyEvent() override;

  EventManager<void(uint32_t)> &CharEvent() override;

  EventManager<void(int, const char **)> &DropEvent() override;

 private:
  bool invalid_{true};

  IMetalQueue *queue_{};
  std::unique_ptr<WindowImGui> imgui_{};
  std::unique_ptr<ImagePresenter> presenter_{};
  bool processing_imgui_{};
  EventManager<void(int, int)> resize_event_{};
  EventManager<void(int, int)> framebufferresize_event_{};
  EventManager<void(bool)> cursorenter_event_{};
  EventManager<void(bool)> focus_event_{};
  EventManager<void(double, double)> mousemove_event_{};
  EventManager<void(int, int, int, double, double)> mousebutton_event_{};
  EventManager<void(const MagnifyGesture &)> magnify_event_;
#if defined(__APPLE__)
  void *magnify_monitor_{};
#endif
  EventManager<void(double, double)> scroll_event_{};
  EventManager<void(int, int, int, int)> key_event_{};
  EventManager<void(uint32_t)> char_event_{};
  EventManager<void(int, const char **)> drop_event_{};
  PresentMode requested_mode_{PRESENT_MODE_VSYNC};
  ColorSpace requested_color_space_{COLOR_SPACE_SRGB_NONLINEAR};
  ColorSpace color_space_{COLOR_SPACE_SRGB_NONLINEAR};
  GLFWwindow *window_{};
  std::mutex next_image_mutex_{};
  ObjectPtr<IMetalImage> next_image_{};
};
}  // namespace grassland::rhi::backend::metal
