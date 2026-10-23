#pragma once
#include "grassland/rhi/rhi.h"
#include "grassland/util/event_manager.h"
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
  static std::unique_ptr<WindowImGui> Create(Device *device, GLFWwindow *window);
  ~WindowImGui();

  ImGuiContext *Context() const {
    return context_;
  }

  bool Record(ICommandContext *commands, IImage *image, EventManager<void()> &callbacks);

 private:
  explicit WindowImGui(Device *device) : device_(device) {
  }

  bool Initialize(GLFWwindow *window);
  bool Render(ICommandContext *commands, IImage *image, ImDrawData *data);
  Device *device_;
  ImGuiContext *context_{};
  bool platform_initialized_{};
  ObjPtr<IImage> font_;
  ObjPtr<ISampler> sampler_;
  ObjPtr<IProgram> program_;
};
}  // namespace grassland::rhi::backend
