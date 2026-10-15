#pragma once
#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/rhi_window.h"

namespace grassland::rhi::backend {

class Window : public Object<rhi::IWindow> {
 public:
  Window();
  ~Window() override;
  rhi::IImage *AcquireFrame() override;
  bool SetPresentMode(PresentMode mode) override;
  GLFWwindow *GLFWWindow() const override;
  bool ShouldClose() const override;
  bool IsKeyDown(int key) const override;
  bool SetTitle(const std::string &title) override;
  // An acquired Image carries one frame lease until all recorded/GPU uses end.
  void AcquireFrameLease();
  void ReleaseFrameLease();
  bool HasFrameLeases() const;
  virtual bool OwnsFrame(const Image *image) const = 0;
  virtual ObjectPtr<Image> AcquireNativeFrame() = 0;
  virtual bool SetNativePresentMode(PresentMode mode) = 0;
  virtual PresentMode GetPresentMode() const = 0;

 protected:
  bool InitializeWindow(const WindowDesc &desc);

 private:
  GLFWwindow *window_{};
  std::atomic<uint32_t> frame_leases_{};
};

}  // namespace grassland::rhi::backend
