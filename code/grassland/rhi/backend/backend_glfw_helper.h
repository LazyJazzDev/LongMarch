#pragma once
#include "grassland/rhi/backend/backend_util.h"

struct GLFWwindow;

namespace grassland::rhi::backend {

namespace glfw {
// Both borrowed interfaces must belong to the same window.
void RegisterWindow(IWindow *window, IBackendWindow *backend);
void UnregisterWindow(GLFWwindow *window);
bool Initialize();
bool PollEvents();
bool IsWindowThread();
GLFWwindow *CreateGLFWWindow(int width, int height, const char *title);
void DestroyWindow(GLFWwindow *window);
#if defined(__APPLE__)
void DestroyCocoaWindow(GLFWwindow *window);
#endif
bool SetWindowTitle(GLFWwindow *window, const char *title);
}  // namespace glfw

}  // namespace grassland::rhi::backend
