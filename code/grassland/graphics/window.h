#pragma once
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {

class Window {
 public:
  Window(int width,
         int height,
         const std::string &title,
         bool fullscreen,
         bool resizable);
  virtual ~Window();

  GLFWwindow *GLFWWindow() const {
    return window_;
  }

  int GetWidth() const;

  int GetHeight() const;

  void SetTitle(const std::string &title);

  void Resize(int new_width, int new_height);

  virtual void CloseWindow();

  bool ShouldClose() const;

  EventManager<void(int, int)> &ResizeEvent() {
    return resize_event_;
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
  // Resize, mouse, keyboard, etc.
  EventManager<void(int, int)> resize_event_;
  EventManager<void(double, double)> mouse_move_event_;
  EventManager<void(int, int, int, double, double)> mouse_button_event_;
  EventManager<void(double, double)> scroll_event_;
  EventManager<void(int, int, int, int)> key_event_;
  EventManager<void(uint32_t)> char_event_;
  EventManager<void(int, const char **)> drop_event_;
};

}  // namespace grassland::graphics
