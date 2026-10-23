#include "grassland/rhi/backend/backend_glfw_helper.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/backend_window.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace grassland::rhi::backend {

namespace glfw {
namespace {
struct State {
  std::thread::id thread;
  bool initialized{};
  std::mutex mutex;
  std::vector<std::pair<GLFWwindow *, std::unique_ptr<WaylandColor>>> destroyed;
  std::unordered_map<GLFWwindow *, Window *> windows;
  bool resize_failed{};

  ~State() {
    if (initialized && thread == std::this_thread::get_id()) {
      // Destroy observers while the GLFW-owned Wayland connection is still alive.
      destroyed.clear();
      glfwTerminate();
    }
  }
};

State &Get() {
  static State state;
  return state;
}
}  // namespace

void RegisterWindow(Window *window) {
  auto native = window->GLFWWindow();
  {
    std::lock_guard<std::mutex> lock(Get().mutex);
    Get().windows.emplace(native, window);
  }
  glfwSetFramebufferSizeCallback(native, [](GLFWwindow *native, int, int) {
    // Native destructors unregister before teardown, under this same lock.
    std::lock_guard<std::mutex> lock(Get().mutex);
    auto it = Get().windows.find(native);
    if (it == Get().windows.end())
      return;
    if (!it->second->RebuildSwapChain())
      Get().resize_failed = true;
  });
  glfwSetWindowIconifyCallback(native, [](GLFWwindow *native, int iconified) {
    // Restoring a Cocoa window need not change its framebuffer dimensions.
    // A mode change while minimized may have discarded its prepared image.
    if (iconified)
      return;
    std::lock_guard<std::mutex> lock(Get().mutex);
    auto it = Get().windows.find(native);
    if (it != Get().windows.end() && !it->second->RebuildSwapChain())
      Get().resize_failed = true;
  });
}

void UnregisterWindow(GLFWwindow *window) {
  std::lock_guard<std::mutex> lock(Get().mutex);
  Get().windows.erase(window);
}

bool Initialize() {
  auto &state = Get();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.initialized) {
#if !defined(NDEBUG)
    return state.thread == std::this_thread::get_id();
#else
    return true;
#endif
  }
  bool initialized = false;
#if defined(__linux__) && defined(GLFW_PLATFORM)
  const char *platform = std::getenv("LONGMARCH_WINDOW_SYSTEM");
  if (platform && std::strcmp(platform, "auto") != 0) {
    int requested;
    if (std::strcmp(platform, "wayland") == 0)
      requested = GLFW_PLATFORM_WAYLAND;
    else if (std::strcmp(platform, "x11") == 0)
      requested = GLFW_PLATFORM_X11;
    else {
      ReportError(-1, "LONGMARCH_WINDOW_SYSTEM must be auto, wayland or x11.");
      return false;
    }
    if (!glfwPlatformSupported(requested)) {
      ReportError(-1, "Requested window system was not compiled into GLFW.");
      return false;
    }
    glfwInitHint(GLFW_PLATFORM, requested);
    initialized = glfwInit() == GLFW_TRUE;
  } else {
    if (glfwPlatformSupported(GLFW_PLATFORM_WAYLAND)) {
      glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);
      initialized = glfwInit() == GLFW_TRUE;
      if (!initialized)
        glfwGetError(nullptr);
    }
    if (!initialized && glfwPlatformSupported(GLFW_PLATFORM_X11)) {
      glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
      initialized = glfwInit() == GLFW_TRUE;
    }
  }
#else
  initialized = glfwInit() == GLFW_TRUE;
#endif
  if (!initialized) {
    const char *message{};
    const int code = glfwGetError(&message);
    ReportError(code, message ? message : "GLFW initialization failed.");
    return false;
  }
  state.thread = std::this_thread::get_id();
  state.initialized = true;
  return true;
}

bool IsWindowThread() {
  return Get().thread == std::this_thread::get_id();
}

bool PollEvents() {
  if (!Initialize())
    return false;
  std::vector<std::pair<GLFWwindow *, std::unique_ptr<WaylandColor>>> destroyed;
  {
    std::lock_guard<std::mutex> lock(Get().mutex);
    destroyed.swap(Get().destroyed);
  }
  for (auto &entry : destroyed) {
    entry.second.reset();
    glfwDestroyWindow(entry.first);
  }
  glfwPollEvents();
  std::lock_guard<std::mutex> lock(Get().mutex);
  return !Get().resize_failed;
}

GLFWwindow *CreateGLFWWindow(int width, int height, const char *title) {
#if !defined(NDEBUG)
  if (width <= 0 || height <= 0) {
    ReportError(-1, "Window dimensions must be positive.");
    return nullptr;
  }
#endif
  if (!Initialize())
    return nullptr;
  glfwDefaultWindowHints();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  auto window = glfwCreateWindow(width, height, title, nullptr, nullptr);
  if (!window) {
    const char *message{};
    const int code = glfwGetError(&message);
    ReportError(code, message ? message : "GLFW window creation failed.");
  }
  return window;
}

bool SetWindowTitle(GLFWwindow *window, const char *title) {
#if !defined(NDEBUG)
  if (!window || !IsWindowThread())
    return false;
#endif
  glfwGetError(nullptr);
  glfwSetWindowTitle(window, title);
  const char *message{};
  const int error = glfwGetError(&message);
  if (error != GLFW_NO_ERROR) {
    ReportError(error, message ? message : "Setting window title failed.");
    return false;
  }
  return true;
}

void DestroyWindow(GLFWwindow *window, std::unique_ptr<WaylandColor> color) {
  if (!window)
    return;
  if (IsWindowThread()) {
    color.reset();
    glfwDestroyWindow(window);
  } else {
    std::lock_guard<std::mutex> lock(Get().mutex);
    Get().destroyed.emplace_back(window, std::move(color));
  }
}
}  // namespace glfw

}  // namespace grassland::rhi::backend
