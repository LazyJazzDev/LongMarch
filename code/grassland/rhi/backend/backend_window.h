#pragma once
#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/backend_image.h"

namespace grassland::rhi::backend {

class IWindow : public IObject {
 public:
  IWindow();
  ~IWindow() override;
  bool ShouldClose() const;
  bool IsKeyDown(Key key) const;
  bool SetTitle(const std::string &title);
  // An acquired Image carries one frame lease until all recorded/GPU uses end.
  void AcquireFrameLease();
  void ReleaseFrameLease();
  bool HasFrameLeases() const;
  virtual ObjectPtr<IImage> AcquireFrame() = 0;
  virtual bool SetPresentMode(PresentMode mode) = 0;
  virtual PresentMode GetPresentMode() const = 0;

 protected:
  bool InitializeWindow(const WindowDesc &desc);
  GLFWwindow *NativeWindow() const;

 private:
  GLFWwindow *window_{};
  std::atomic<uint32_t> frame_leases_{};
};

}  // namespace grassland::rhi::backend
