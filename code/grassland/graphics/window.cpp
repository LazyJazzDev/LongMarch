#include "grassland/graphics/window.h"

namespace grassland::graphics {

namespace {
bool glfw_initialized_{false};
void InitializeGLFW() {
  if (!glfw_initialized_) {
    if (!glfwInit()) {
      throw std::runtime_error("Failed to initialize GLFW");
    }
    glfw_initialized_ = true;
  }
}
}  // namespace

Window::Window(int width, int height, const std::string &title) {
  InitializeGLFW();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);

  if (!window_) {
    throw std::runtime_error("Failed to create GLFW window");
  }
}

Window::~Window() {
  CloseWindow();
}

int Window::GetWidth() const {
  int width, height;
  glfwGetWindowSize(window_, &width, &height);
  return width;
}

int Window::GetHeight() const {
  int width, height;
  glfwGetWindowSize(window_, &width, &height);
  return height;
}

void Window::SetTitle(const std::string &title) {
  glfwSetWindowTitle(window_, title.c_str());
}

void Window::Resize(int new_width, int new_height) {
  glfwSetWindowSize(window_, new_width, new_height);
}

void Window::CloseWindow() {
  glfwDestroyWindow(window_);
  window_ = nullptr;
}

bool Window::ShouldClose() const {
  return glfwWindowShouldClose(window_);
}

}  // namespace grassland::graphics
