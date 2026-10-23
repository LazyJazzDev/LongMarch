#pragma once
#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>

#include "grassland/rhi/rhi_object.h"
#include "grassland/util/event_manager.h"

struct ImGuiContext;

namespace grassland::rhi {
class IImage;
class ICommandContext;
// Window operations and presentation run on the main thread.
bool PollEvents();

class IWindow : public IObject {
 public:
  // Registries belong to the window; callbacks run on the window thread.
  // ImGui callbacks run with this window's context between NewFrame and Render
  // when CmdPresent records the overlay. Do not present/reconfigure the window
  // recursively from these callbacks. Input events are delivered even when ImGui
  // captures input; inspect WantCaptureMouse/WantCaptureKeyboard in its context.
  virtual EventManager<void(int, int)> &ResizeEvent() = 0;
  virtual EventManager<void(int, int)> &FramebufferResizeEvent() = 0;
  virtual EventManager<void(bool)> &CursorEnterEvent() = 0;
  virtual EventManager<void(bool)> &FocusEvent() = 0;
  virtual EventManager<void(double, double)> &MouseMoveEvent() = 0;
  virtual EventManager<void(int, int, int, double, double)> &MouseButtonEvent() = 0;
  virtual EventManager<void(double, double)> &ScrollEvent() = 0;
  virtual EventManager<void(int, int, int, int)> &KeyEvent() = 0;
  virtual EventManager<void(uint32_t)> &CharEvent() = 0;
  virtual EventManager<void(int, const char **)> &DropEvent() = 0;
  virtual EventManager<void()> &ImGuiEvent() = 0;

  // Creates a separate ImGui context and font texture. Disabling releases them.
  // Completed/in-flight recordings own the GPU resources they use independently.
  // Font options apply only when creating the context. nullptr uses the built-in
  // font; file fonts include the ChineseFull glyph range, matching graphics.
  // Repeated enable calls keep the existing font. Disable first to reinitialize.
  virtual bool EnableImGui(bool enabled = true, const char *font_file_path = nullptr, float font_size = 13.0f) = 0;
  virtual ImGuiContext *GetImGuiContext() const = 0;

  // Borrowed handle; Window owns it. Use GLFW operations on the main thread.
  virtual GLFWwindow *GLFWWindow() const = 0;
  virtual bool ShouldClose() const = 0;
  virtual bool IsKeyDown(int key) const = 0;
  virtual bool SetTitle(const std::string &title) = 0;
  // Returns an owned reference to the same next image until Present submission
  // or swapchain replacement. Null while minimized, zero-sized or invalid.
  // This getter never acquires a native image or rebuilds the swapchain.
  // Release caller-held frames and recordings before swapchain replacement or
  // Window destruction. Window waits for GPU use; it is not retained by Device.
  virtual IImage *GetNextImage() = 0;
  // Known swapchain invalidation, not a native surface probe. OUT_OF_DATE
  // marks this state without rebuilding during command submission.
  virtual bool IsSwapChainInvalid() const = 0;
  // Call on the window thread outside recording, after releasing caller-held
  // frames and recordings. Success may remain invalid if the surface changes
  // again or is minimized; check IsSwapChainInvalid() before rendering.
  virtual bool RebuildSwapChain() = 0;
  virtual bool SetPresentMode(PresentMode mode) = 0;
  virtual PresentMode GetPresentMode() const = 0;
  // Currently implemented combinations supported by this window's surface.
  // Formats are selected automatically at creation/rebuild. Empty on failure.
  virtual std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const = 0;

 protected:
  ~IWindow() override;
};

}  // namespace grassland::rhi
