#include "demo_window.h"

namespace demo {

Window::Window(grassland::rhi::IWindow *window) : window_(window) {
}

Window::~Window() = default;

glm::ivec2 Window::GetSize() const {
  glm::ivec2 size{};
  glfwGetWindowSize(GLFWWindow(), &size.x, &size.y);
  return size;
}

glm::ivec2 Window::GetFramebufferSize() const {
  glm::ivec2 size{};
  glfwGetFramebufferSize(GLFWWindow(), &size.x, &size.y);
  return size;
}

glm::dvec2 Window::GetCursorPosition() const {
  glm::dvec2 position{};
  glfwGetCursorPos(GLFWWindow(), &position.x, &position.y);
  return position;
}

}  // namespace demo
