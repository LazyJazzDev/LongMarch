#pragma once
#include <mutex>

#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/rhi_window.h"

struct ImGuiContext;

namespace grassland::rhi::backend {

class WindowImGui;
class Device;

class Window : public Object<rhi::IWindow> {
 public:
  Window();
  bool EnableImGui(bool enabled = true, const char *font_file_path = nullptr, float font_size = 13.0f) override;
  bool UpdateImGui(const std::function<void()> &callback) override;
  ImGuiContext *GetImGuiContext() const;
  bool RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image);

  EventManager<void(int, int)> &ResizeEvent() override {
    return resize_event_;
  }

  EventManager<void(int, int)> &FramebufferResizeEvent() override {
    return framebufferresize_event_;
  }

  EventManager<void(bool)> &CursorEnterEvent() override {
    return cursorenter_event_;
  }

  EventManager<void(bool)> &FocusEvent() override {
    return focus_event_;
  }

  EventManager<void(double, double)> &MouseMoveEvent() override {
    return mousemove_event_;
  }

  EventManager<void(int, int, int, double, double)> &MouseButtonEvent() override {
    return mousebutton_event_;
  }

  EventManager<void(double, double)> &ScrollEvent() override {
    return scroll_event_;
  }

  EventManager<void(int, int, int, int)> &KeyEvent() override {
    return key_event_;
  }

  EventManager<void(uint32_t)> &CharEvent() override {
    return char_event_;
  }

  EventManager<void(int, const char **)> &DropEvent() override {
    return drop_event_;
  }

  ~Window() override;
  rhi::IImage *GetNextImage() override;
  bool IsSwapChainInvalid() const override;
  bool RebuildSwapChain() override;
  virtual bool UpdateNextImage();
  void ReleaseNextImage();
  bool SetPresentMode(PresentMode mode) override;
  bool SetColorSpace(ColorSpace color_space) override;
  ColorSpace GetColorSpace() const override;
  bool GetHDRReferenceWhiteNits(float &nits) const override;
  float GetHDRReferenceWhiteScale() const override;
  float GetHDRHeadroom() const override;
  GLFWwindow *GLFWWindow() const override;
  bool ShouldClose() const override;
  bool IsKeyDown(int key) const override;
  bool SetTitle(const std::string &title) override;
  virtual ObjectPtr<Image> AcquireNativeFrame() = 0;
  virtual bool RebuildNativeSwapChain(PresentMode mode) = 0;

 protected:
  bool InitializeWindow(Device *device, const WindowDesc &desc);
  bool invalid_{true};
  ColorSpace RequestedColorSpace() const;

 private:
  Device *device_{};
  std::unique_ptr<WindowImGui> imgui_;
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
