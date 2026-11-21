#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_imgui.h"
#include "imgui.h"

namespace grassland::rhi::backend {
class ImGuiContextScope final {
 public:
  explicit ImGuiContextScope(ImGuiContext *context);
  ~ImGuiContextScope();

 private:
  ImGuiContext *previous_;
};

class WindowImGui final {
 public:
  static std::unique_ptr<WindowImGui> Create(IDevice *device,
                                             GLFWwindow *window,
                                             const char *font_file_path,
                                             float font_size);
  ~WindowImGui();
  ImGuiContext *Context() const;
  bool Update(const std::function<void()> &callback);
  bool Record(ICommandContext *commands,
              IImage *image,
              ColorSpace color_space = COLOR_SPACE_SRGB_NONLINEAR,
              float reference_white_scale = 1.0f);

 private:
  explicit WindowImGui(IDevice *device);
  bool Initialize(GLFWwindow *window, const char *font_file_path, float font_size);
  IDevice *device_;
  std::unique_ptr<ImGuiRenderer> renderer_;
  bool platform_initialized_{};
};
}  // namespace grassland::rhi::backend
