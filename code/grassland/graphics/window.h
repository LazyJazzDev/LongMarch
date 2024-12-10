#pragma once
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {

class Window {
 public:
  Window(int width, int height, const std::string &title);
  virtual ~Window() = default;

  GLFWwindow *GLFWWindow() const {
    return window_;
  }

  int GetWidth() const;

  int GetHeight() const;

  void SetTitle(const std::string &title);

  void Resize(int new_width, int new_height);

  virtual void CloseWindow();

  bool ShouldClose() const;

 private:
  GLFWwindow *window_;
};

}  // namespace grassland::graphics
