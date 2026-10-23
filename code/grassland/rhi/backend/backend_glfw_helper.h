#pragma once
#include "grassland/rhi/backend/backend_wayland_color.h"

struct GLFWwindow;

namespace grassland::rhi::backend {

class Window;

namespace glfw {
void RegisterWindow(Window *window);
void UnregisterWindow(GLFWwindow *window);
bool Initialize();
bool PollEvents();
bool IsWindowThread();
GLFWwindow *CreateGLFWWindow(int width, int height, const char *title);
void DestroyWindow(GLFWwindow *window, std::unique_ptr<WaylandColor> color);
bool SetWindowTitle(GLFWwindow *window, const char *title);
}  // namespace glfw

}  // namespace grassland::rhi::backend
