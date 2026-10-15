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
  bool RebuildSwapChain();
  virtual bool UpdateNextImage();
  void ReleaseNextImage();
  bool SetPresentMode(PresentMode mode) override;
  GLFWwindow *GLFWWindow() const override;
  bool ShouldClose() const override;
  bool IsKeyDown(int key) const override;
  bool SetTitle(const std::string &title) override;
  virtual ObjectPtr<Image> AcquireNativeFrame() = 0;
  virtual bool RebuildNativeSwapChain(PresentMode mode) = 0;

 protected:
  bool InitializeWindow(const WindowDesc &desc);

 private:
  PresentMode requested_mode_{PRESENT_MODE_VSYNC};
  GLFWwindow *window_{};
  std::mutex next_image_mutex_;
  ObjectPtr<Image> next_image_;
};

}  // namespace grassland::rhi::backend
