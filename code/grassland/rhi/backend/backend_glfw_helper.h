#pragma once
#include "grassland/rhi/backend/backend_util.h"

struct GLFWwindow;
#ifdef __OBJC__
@class NSEvent;
#endif
namespace grassland::rhi {
struct MagnifyGesture;
}

namespace grassland::rhi::backend {

namespace glfw {
// Both borrowed interfaces must belong to the same window.
void RegisterWindow(IWindow *window, IBackendWindow *backend);
void UnregisterWindow(GLFWwindow *window);
// Look up and retain the registered window while invoking its callbacks.
void DispatchMagnifyEvent(GLFWwindow *window, const MagnifyGesture &gesture);
bool Initialize();
bool PollEvents();
bool IsWindowThread();
GLFWwindow *CreateGLFWWindow(int width, int height, const char *title);
void DestroyWindow(GLFWwindow *window);
#if defined(__APPLE__)
void DestroyCocoaWindow(GLFWwindow *window);
void *InstallMagnifyEvents(GLFWwindow *window);
void RemoveMagnifyEvents(void *monitor);
#ifdef __OBJC__
void DispatchCocoaMagnifyEvent(GLFWwindow *window, NSEvent *event);
#endif
#endif
bool SetWindowTitle(GLFWwindow *window, const char *title);
}  // namespace glfw

}  // namespace grassland::rhi::backend
