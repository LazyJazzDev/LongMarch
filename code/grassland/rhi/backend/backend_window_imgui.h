#pragma once
#include "grassland/rhi/rhi.h"
#include "imgui.h"

namespace grassland::rhi::backend {
class Device;

// All ImGui and GLFW operations run on the window thread. Restore the caller's
// context, including when routing input to a different window.
class ImGuiContextScope {
 public:
  explicit ImGuiContextScope(ImGuiContext *context) : previous_(ImGui::GetCurrentContext()) {
    ImGui::SetCurrentContext(context);
  }

  ~ImGuiContextScope() {
    ImGui::SetCurrentContext(previous_);
  }

 private:
  ImGuiContext *previous_;
};

class WindowImGui {
 public:
  static std::unique_ptr<WindowImGui> Create(Device *device,
                                             GLFWwindow *window,
                                             const char *font_file_path,
                                             float font_size);
  ~WindowImGui();

  ImGuiContext *Context() const {
    return context_;
  }

  bool Update(const std::function<void()> &callback);
  bool Record(ICommandContext *commands,
              IImage *image,
              ColorSpace color_space = COLOR_SPACE_SRGB_NONLINEAR,
              float reference_white_scale = 1.0f);

 private:
  explicit WindowImGui(Device *device) : device_(device) {
  }

  bool Initialize(GLFWwindow *window, const char *font_file_path, float font_size);
  bool Render(ICommandContext *commands, IImage *image, ImDrawData *data);
  bool ConvertPresentation(ICommandContext *commands,
                           IImage *source,
                           IImage *target,
                           ColorSpace color_space,
                           float reference_white_scale,
                           bool encode);
  Device *device_;
  ImGuiContext *context_{};
  bool platform_initialized_{};
  bool draw_data_ready_{};
  ObjPtr<IImage> font_;
  ObjPtr<ISampler> sampler_;
  ObjPtr<IProgram> program_;
  ObjPtr<IBuffer> vertex_buffer_;
  ObjPtr<IBuffer> index_buffer_;
  ObjPtr<IImage> composition_;
  ObjPtr<IBuffer> display_settings_;
  ObjPtr<IProgram> decode_program_;
  ObjPtr<IProgram> encode_program_;
  ColorSpace conversion_space_{COLOR_SPACE_SRGB_NONLINEAR};
};
}  // namespace grassland::rhi::backend
