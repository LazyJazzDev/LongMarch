#pragma once
#include <mutex>

#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/rhi_window.h"

struct ImGuiContext;

namespace grassland::rhi::backend {

class WindowImGui;
class ImagePresenter;
class Device;

class Window : public Object<rhi::IWindow> {
 public:
  Window();
  bool ConfigureImGui(bool enabled = true, const char *font_file_path = nullptr, float font_size = 13.0f);
  bool RunImGuiFrame(const std::function<void()> &callback);
  ImGuiContext *GetImGuiContext() const;
  bool RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image);
  bool RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image, ColorSpace color_space, float white_scale);
  bool RecordPresentImage(ICommandContext *commands, IImage *image, ColorSpace color_space);
  const void *DeviceIdentity() const;

  EventManager<void(int, int)> &ResizeCallbacks() {
    return resize_event_;
  }

  EventManager<void(int, int)> &FramebufferResizeCallbacks() {
    return framebufferresize_event_;
  }

  EventManager<void(bool)> &CursorEnterCallbacks() {
    return cursorenter_event_;
  }

  EventManager<void(bool)> &FocusCallbacks() {
    return focus_event_;
  }

  EventManager<void(double, double)> &MouseMoveCallbacks() {
    return mousemove_event_;
  }

  EventManager<void(int, int, int, double, double)> &MouseButtonCallbacks() {
    return mousebutton_event_;
  }

  EventManager<void(double, double)> &ScrollCallbacks() {
    return scroll_event_;
  }

  EventManager<void(int, int, int, int)> &KeyCallbacks() {
    return key_event_;
  }

  EventManager<void(uint32_t)> &CharCallbacks() {
    return char_event_;
  }

  EventManager<void(int, const char **)> &DropCallbacks() {
    return drop_event_;
  }

  ~Window() override;
  rhi::IImage *PreparedImage();
  bool SwapChainNeedsRebuild() const;
  bool RecreateSwapChain();
  virtual bool UpdateNextImage();
  void ReleaseNextImage();
  bool ChangePresentMode(PresentMode mode);
  bool ChangeColorSpace(ColorSpace color_space);
  ColorSpace ActiveColorSpace() const;
  bool QueryHDRReferenceWhiteNits(float &nits) const;
  float HDRReferenceWhiteScale() const;
  float HDRHeadroom() const;
  GLFWwindow *NativeWindow() const;
  bool WindowShouldClose() const;
  bool KeyPressed(int key) const;
  bool ChangeTitle(const std::string &title);
  virtual ObjectPtr<Image> AcquireNativeFrame() = 0;
  virtual bool RebuildNativeSwapChain(PresentMode mode) = 0;

 protected:
  bool InitializeWindow(Device *device, const WindowDesc &desc);
  bool invalid_{true};
  ColorSpace RequestedColorSpace() const;

 private:
  Device *device_{};
  std::unique_ptr<WindowImGui> imgui_;
  std::unique_ptr<ImagePresenter> presenter_;
  bool processing_imgui_{};
  EventManager<void(int, int)> resize_event_;
  EventManager<void(int, int)> framebufferresize_event_;
  EventManager<void(bool)> cursorenter_event_;
  EventManager<void(bool)> focus_event_;
  EventManager<void(double, double)> mousemove_event_;
  EventManager<void(int, int, int, double, double)> mousebutton_event_;
  EventManager<void(double, double)> scroll_event_;
  EventManager<void(int, int, int, int)> key_event_;
  EventManager<void(uint32_t)> char_event_;
  EventManager<void(int, const char **)> drop_event_;
  PresentMode requested_mode_{PRESENT_MODE_VSYNC};
  ColorSpace requested_color_space_{COLOR_SPACE_SRGB_NONLINEAR};
  ColorSpace color_space_{COLOR_SPACE_SRGB_NONLINEAR};
  GLFWwindow *window_{};
  std::mutex next_image_mutex_;
  ObjectPtr<Image> next_image_;
};

}  // namespace grassland::rhi::backend
