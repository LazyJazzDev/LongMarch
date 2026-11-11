#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend {
class Window;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::Window)
#include <mutex>

#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/rhi_window.h"

struct ImGuiContext;

namespace grassland::rhi::backend {

class WindowImGui;
class ImagePresenter;
class Device;

struct WindowState;

class Window : public rhi::IWindow {
 protected:
  void InitializeWindow();

 public:
  Queue *OwnerQueue() const;
  bool EnableImGui(bool enabled = true, const char *font_file_path = nullptr, float font_size = 13.0f) override;
  bool UpdateImGui(const std::function<void()> &callback) override;
  ImGuiContext *GetImGuiContext() const;
  bool RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image);
  bool RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image, ColorSpace color_space, float white_scale);
  bool RecordPresentImage(ICommandContext *commands, IImage *image, ColorSpace color_space);
  const void *DeviceIdentity() const;
  EventManager<void(int, int)> &ResizeEvent() override;
  EventManager<void(int, int)> &FramebufferResizeEvent() override;
  EventManager<void(bool)> &CursorEnterEvent() override;
  EventManager<void(bool)> &FocusEvent() override;
  EventManager<void(double, double)> &MouseMoveEvent() override;
  EventManager<void(int, int, int, double, double)> &MouseButtonEvent() override;
  EventManager<void(double, double)> &ScrollEvent() override;
  EventManager<void(int, int, int, int)> &KeyEvent() override;
  EventManager<void(uint32_t)> &CharEvent() override;
  EventManager<void(int, const char **)> &DropEvent() override;

 protected:
  void DestroyWindow();

 public:
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
  ColorSpace RequestedColorSpace() const;
  virtual WindowState &StateWindow() = 0;
  virtual const WindowState &StateWindow() const = 0;
};

}  // namespace grassland::rhi::backend
