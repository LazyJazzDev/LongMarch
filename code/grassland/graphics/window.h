#pragma once
#include "grassland/graphics/graphics_util.h"
#include "imgui.h"

namespace grassland::graphics {

enum class MagnifyPhase { kBegin, kUpdate, kEnd, kCancel };

// Incremental scale (1.0 = unchanged), with a focus in GLFW window coordinates:
// logical units, origin at the top-left of the content area. Cancellation ends
// the gesture without undoing previously delivered increments.
struct MagnifyGesture {
  double scale{1.0};
  double x{};
  double y{};
  MagnifyPhase phase{MagnifyPhase::kUpdate};
};

class Window {
 public:
  Window(int width, int height, const std::string &title, bool fullscreen, bool resizable, bool enable_hdr);
  virtual ~Window();

  GLFWwindow *GLFWWindow() const {
    return window_;
  }

  int GetWidth() const;

  int GetHeight() const;

  // Window size and cursor position use logical content coordinates; the
  // framebuffer size uses physical pixels and can be zero when minimized.
  glm::ivec2 GetSize() const;
  glm::ivec2 GetFramebufferSize() const;
  glm::dvec2 GetCursorPosition() const;
  bool IsKeyDown(int key) const;
  bool IsMouseButtonDown(int button) const;
  bool IsFocused() const;
  void Focus();
  void RequestClose();

  // Desktop coordinates and decoration widths use logical screen units.
  glm::ivec2 GetPosition() const;
  void SetPosition(int x, int y);
  glm::ivec4 GetFrameSize() const;        // left, top, right, bottom
  glm::ivec4 GetMonitorWorkArea() const;  // x, y, width, height; largest overlap

  // Process events for all windows on the main thread.
  static void PollEvents();

  void SetTitle(const std::string &title);

  std::string GetTitle() const;

  void Resize(int new_width, int new_height);

  virtual void CloseWindow();

  bool ShouldClose() const;

  virtual void SetHDR(bool enable_hdr);

  virtual void InitImGui(const char *font_file_path = nullptr, float font_size = 13.0f) = 0;
  virtual void TerminateImGui() = 0;
  virtual void BeginImGuiFrame() = 0;
  virtual void EndImGuiFrame() = 0;
  virtual ImGuiContext *GetImGuiContext() const = 0;

  EventManager<void(int, int)> &ResizeEvent() {
    return resize_event_;
  }

  EventManager<void(int, int)> &FramebufferResizeEvent() {
    return framebuffer_resize_event_;
  }

  EventManager<void(bool)> &CursorEnterEvent() {
    return cursor_enter_event_;
  }

  EventManager<void(bool)> &FocusEvent() {
    return focus_event_;
  }

  EventManager<void(double, double)> &MouseMoveEvent() {
    return mouse_move_event_;
  }

  EventManager<void(int, int, int, double, double)> &MouseButtonEvent() {
    return mouse_button_event_;
  }

  EventManager<void(double, double)> &ScrollEvent() {
    return scroll_event_;
  }

  // Native gesture support for this window/platform, not device availability.
  // Unsupported platforms emit no magnify events. Ctrl + scroll stays a separate
  // input path: driver-emulated scroll is never also synthesized as magnification.
  bool SupportsMagnifyGestures() const {
    return magnify_monitor_ != nullptr;
  }

  // Dispatched on the window event thread, just like ScrollEvent().
  EventManager<void(const MagnifyGesture &)> &MagnifyEvent() {
    return magnify_event_;
  }

  EventManager<void(int, int, int, int)> &KeyEvent() {
    return key_event_;
  }

  EventManager<void(uint32_t)> &CharEvent() {
    return char_event_;
  }

  EventManager<void(int, const char **)> &DropEvent() {
    return drop_event_;
  }

 private:
  GLFWwindow *window_;
  void *magnify_monitor_{};
  EventManager<void(const MagnifyGesture &)> magnify_event_;
  // Resize, mouse, keyboard, etc.
  EventManager<void(int, int)> resize_event_;
  EventManager<void(int, int)> framebuffer_resize_event_;
  EventManager<void(bool)> cursor_enter_event_;
  EventManager<void(bool)> focus_event_;
  EventManager<void(double, double)> mouse_move_event_;
  EventManager<void(int, int, int, double, double)> mouse_button_event_;
  EventManager<void(double, double)> scroll_event_;
  EventManager<void(int, int, int, int)> key_event_;
  EventManager<void(uint32_t)> char_event_;
  EventManager<void(int, const char **)> drop_event_;

 protected:
  bool enable_hdr_;

 public:
#if defined(LONGMARCH_PYTHON_ENABLED)
  static void PybindClassRegistration(py::classh<Window> &c);
#endif
};

}  // namespace grassland::graphics
