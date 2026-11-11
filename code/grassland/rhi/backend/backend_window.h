#pragma once
#include <mutex>

#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_window.h"

struct ImGuiContext;

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::IBackendWindow)

namespace grassland::rhi::backend {
#ifdef __APPLE__
float CocoaHDRHeadroom(GLFWwindow *window);
#endif

class WindowImGui;
class ImagePresenter;

class IBackendWindow : public rhi::IWindow {
 public:
  virtual IBackendQueue *OwnerQueue() const = 0;

  bool EnableImGui(bool enabled = true, const char *font_file_path = nullptr, float font_size = 13.0f) override = 0;

  bool UpdateImGui(const std::function<void()> &callback) override = 0;

  virtual ImGuiContext *GetImGuiContext() const = 0;

  virtual bool RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image) = 0;

  virtual bool RecordImGui(rhi::ICommandContext *commands,
                           rhi::IImage *image,
                           ColorSpace color_space,
                           float white_scale) = 0;

  virtual bool RecordPresentImage(ICommandContext *commands, IImage *image, ColorSpace color_space) = 0;

  virtual const void *DeviceIdentity() const = 0;

  EventManager<void(int, int)> &ResizeEvent() override = 0;

  EventManager<void(int, int)> &FramebufferResizeEvent() override = 0;

  EventManager<void(bool)> &CursorEnterEvent() override = 0;

  EventManager<void(bool)> &FocusEvent() override = 0;

  EventManager<void(double, double)> &MouseMoveEvent() override = 0;

  EventManager<void(int, int, int, double, double)> &MouseButtonEvent() override = 0;

  EventManager<void(double, double)> &ScrollEvent() override = 0;

  EventManager<void(int, int, int, int)> &KeyEvent() override = 0;

  EventManager<void(uint32_t)> &CharEvent() override = 0;

  EventManager<void(int, const char **)> &DropEvent() override = 0;

  rhi::IImage *GetNextImage() override = 0;

  bool IsSwapChainInvalid() const override = 0;

  bool RebuildSwapChain() override = 0;

  virtual bool UpdateNextImage() = 0;

  virtual void ReleaseNextImage() = 0;

  bool SetPresentMode(PresentMode mode) override = 0;

  bool SetColorSpace(ColorSpace color_space) override = 0;

  ColorSpace GetColorSpace() const override = 0;

  bool GetHDRReferenceWhiteNits(float &nits) const override = 0;

  float GetHDRReferenceWhiteScale() const override = 0;

  float GetHDRHeadroom() const override = 0;

  GLFWwindow *GLFWWindow() const override = 0;

  bool ShouldClose() const override = 0;

  bool IsKeyDown(int key) const override = 0;

  bool SetTitle(const std::string &title) override = 0;

  virtual ObjectPtr<IBackendImage> AcquireNativeFrame() = 0;

  virtual bool RebuildNativeSwapChain(PresentMode mode) = 0;
};

}  // namespace grassland::rhi::backend
