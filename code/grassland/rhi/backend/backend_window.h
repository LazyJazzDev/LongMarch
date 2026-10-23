#pragma once
#include <mutex>

#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/rhi_window.h"

namespace grassland::rhi::backend {

class Window : public Object<rhi::IWindow> {
 public:
  Window();
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
  GLFWwindow *GLFWWindow() const override;
  bool ShouldClose() const override;
  bool IsKeyDown(int key) const override;
  bool SetTitle(const std::string &title) override;
  virtual ObjectPtr<Image> AcquireNativeFrame() = 0;
  virtual bool RebuildNativeSwapChain(PresentMode mode) = 0;

 protected:
  bool InitializeWindow(const WindowDesc &desc);
  bool invalid_{true};
  ColorSpace RequestedColorSpace() const;

 private:
  PresentMode requested_mode_{PRESENT_MODE_VSYNC};
  ColorSpace requested_color_space_{COLOR_SPACE_SRGB_NONLINEAR};
  ColorSpace color_space_{COLOR_SPACE_SRGB_NONLINEAR};
  GLFWwindow *window_{};
  std::mutex next_image_mutex_;
  ObjectPtr<Image> next_image_;
};

}  // namespace grassland::rhi::backend
