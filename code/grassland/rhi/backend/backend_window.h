#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendWindow)
#include <mutex>

#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/rhi_window.h"

struct ImGuiContext;

namespace grassland::rhi::backend {

class WindowImGui;
class ImagePresenter;
struct WindowState;

class IBackendWindow : public rhi::IWindow {
 protected:
  void InitializeWindow();

 public:
  IBackendQueue *OwnerQueue() const;
  bool EnableImGui(bool enabled = true, const char *font_file_path = nullptr, float font_size = 13.0f) override = 0;
  bool CommonEnableImGui(bool enabled = true, const char *font_file_path = nullptr, float font_size = 13.0f);
  bool UpdateImGui(const std::function<void()> &callback) override = 0;
  bool CommonUpdateImGui(const std::function<void()> &callback);
  ImGuiContext *GetImGuiContext() const;
  bool RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image);
  bool RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image, ColorSpace color_space, float white_scale);
  bool RecordPresentImage(ICommandContext *commands, IImage *image, ColorSpace color_space);
  const void *DeviceIdentity() const;
  EventManager<void(int, int)> &ResizeEvent() override = 0;
  EventManager<void(int, int)> &CommonResizeEvent();
  EventManager<void(int, int)> &FramebufferResizeEvent() override = 0;
  EventManager<void(int, int)> &CommonFramebufferResizeEvent();
  EventManager<void(bool)> &CursorEnterEvent() override = 0;
  EventManager<void(bool)> &CommonCursorEnterEvent();
  EventManager<void(bool)> &FocusEvent() override = 0;
  EventManager<void(bool)> &CommonFocusEvent();
  EventManager<void(double, double)> &MouseMoveEvent() override = 0;
  EventManager<void(double, double)> &CommonMouseMoveEvent();
  EventManager<void(int, int, int, double, double)> &MouseButtonEvent() override = 0;
  EventManager<void(int, int, int, double, double)> &CommonMouseButtonEvent();
  EventManager<void(double, double)> &ScrollEvent() override = 0;
  EventManager<void(double, double)> &CommonScrollEvent();
  EventManager<void(int, int, int, int)> &KeyEvent() override = 0;
  EventManager<void(int, int, int, int)> &CommonKeyEvent();
  EventManager<void(uint32_t)> &CharEvent() override = 0;
  EventManager<void(uint32_t)> &CommonCharEvent();
  EventManager<void(int, const char **)> &DropEvent() override = 0;
  EventManager<void(int, const char **)> &CommonDropEvent();

 protected:
  void DestroyWindow();

 public:
  rhi::IImage *GetNextImage() override = 0;
  rhi::IImage *CommonGetNextImage();
  bool IsSwapChainInvalid() const override = 0;
  bool CommonIsSwapChainInvalid() const;
  bool RebuildSwapChain() override = 0;
  bool CommonRebuildSwapChain();
  virtual bool UpdateNextImage() = 0;
  bool CommonUpdateNextImage();
  void ReleaseNextImage();
  bool SetPresentMode(PresentMode mode) override = 0;
  bool CommonSetPresentMode(PresentMode mode);
  bool SetColorSpace(ColorSpace color_space) override = 0;
  bool CommonSetColorSpace(ColorSpace color_space);
  ColorSpace GetColorSpace() const override = 0;
  ColorSpace CommonGetColorSpace() const;
  bool GetHDRReferenceWhiteNits(float &nits) const override = 0;
  bool CommonGetHDRReferenceWhiteNits(float &nits) const;
  float GetHDRReferenceWhiteScale() const override = 0;
  float CommonGetHDRReferenceWhiteScale() const;
  float GetHDRHeadroom() const override = 0;
  float CommonGetHDRHeadroom() const;
  GLFWwindow *GLFWWindow() const override = 0;
  GLFWwindow *CommonGLFWWindow() const;
  bool ShouldClose() const override = 0;
  bool CommonShouldClose() const;
  bool IsKeyDown(int key) const override = 0;
  bool CommonIsKeyDown(int key) const;
  bool SetTitle(const std::string &title) override = 0;
  bool CommonSetTitle(const std::string &title);
  virtual ObjectPtr<IBackendImage> AcquireNativeFrame() = 0;
  virtual bool RebuildNativeSwapChain(PresentMode mode) = 0;

 protected:
  bool InitializeWindow(IBackendDevice *device, const WindowDesc &desc);
  ColorSpace RequestedColorSpace() const;
  virtual WindowState &StateWindow() = 0;
  virtual const WindowState &StateWindow() const = 0;
};

}  // namespace grassland::rhi::backend
