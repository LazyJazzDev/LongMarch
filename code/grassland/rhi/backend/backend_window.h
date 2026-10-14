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
  virtual ObjectPtr<IImage> AcquireFrame() = 0;
  virtual bool SetPresentMode(PresentMode mode) = 0;
  virtual PresentMode GetPresentMode() const = 0;

 protected:
  bool InitializeWindow(const WindowDesc &desc);
  GLFWwindow *NativeWindow() const;

 private:
  GLFWwindow *window_{};
};

}  // namespace grassland::rhi::backend
