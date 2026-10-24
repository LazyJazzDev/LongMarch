#include "grassland/rhi/backend/backend_glfw_helper.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/backend_window_imgui.h"
#include "imgui_impl_glfw.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace grassland::rhi::backend {

namespace glfw {
namespace {
struct State {
  std::thread::id thread;
  bool initialized{};
  std::mutex mutex;
  std::vector<GLFWwindow *> destroyed;
  std::unordered_map<GLFWwindow *, Window *> windows;
  bool resize_failed{};

  ~State() {
    if (initialized && thread == std::this_thread::get_id())
      glfwTerminate();
  }
};

State &Get() {
  static State state;
  return state;
}

ObjectPtr<Window> FindWindow(GLFWwindow *native) {
  std::lock_guard<std::mutex> lock(Get().mutex);
  const auto it = Get().windows.find(native);
  return it == Get().windows.end() ? ObjectPtr<Window>{} : ObjectPtr<Window>::Retain(it->second);
}

}  // namespace

void RegisterWindow(Window *window) {
  auto native = window->GLFWWindow();
  {
    std::lock_guard<std::mutex> lock(Get().mutex);
    Get().windows.emplace(native, window);
  }
  glfwSetFramebufferSizeCallback(native, [](GLFWwindow *native, int width, int height) {
    auto window = FindWindow(native);
    if (!window)
      return;
    if (!window->RebuildSwapChain())
      Get().resize_failed = true;
    window->FramebufferResizeEvent().InvokeCallbacks(width, height);
  });
  glfwSetWindowIconifyCallback(native, [](GLFWwindow *native, int iconified) {
    auto window = FindWindow(native);
    if (window && !iconified && !window->RebuildSwapChain())
      Get().resize_failed = true;
  });
  glfwSetWindowSizeCallback(native, [](GLFWwindow *native, int width, int height) {
    auto window = FindWindow(native);
    if (!window)
      return;
    window->ResizeEvent().InvokeCallbacks(width, height);
  });
  glfwSetCursorEnterCallback(native, [](GLFWwindow *native, int entered) {
    auto window = FindWindow(native);
    if (!window)
      return;
    if (auto context = window->GetImGuiContext()) {
      ImGuiContextScope scope(context);
      ImGui_ImplGlfw_CursorEnterCallback(native, entered);
    }
    window->CursorEnterEvent().InvokeCallbacks(bool(entered));
  });
  glfwSetWindowFocusCallback(native, [](GLFWwindow *native, int focused) {
    auto window = FindWindow(native);
    if (!window)
      return;
    if (auto context = window->GetImGuiContext()) {
      ImGuiContextScope scope(context);
      ImGui_ImplGlfw_WindowFocusCallback(native, focused);
    }
    window->FocusEvent().InvokeCallbacks(bool(focused));
  });
  glfwSetCursorPosCallback(native, [](GLFWwindow *native, double x, double y) {
    auto window = FindWindow(native);
    if (!window)
      return;
    if (auto context = window->GetImGuiContext()) {
      ImGuiContextScope scope(context);
      ImGui_ImplGlfw_CursorPosCallback(native, x, y);
    }
    window->MouseMoveEvent().InvokeCallbacks(x, y);
  });
  glfwSetMouseButtonCallback(native, [](GLFWwindow *native, int button, int action, int mods) {
    auto window = FindWindow(native);
    if (!window)
      return;
    if (auto context = window->GetImGuiContext()) {
      ImGuiContextScope scope(context);
      ImGui_ImplGlfw_MouseButtonCallback(native, button, action, mods);
    }
    double x, y;
    glfwGetCursorPos(native, &x, &y);
    window->MouseButtonEvent().InvokeCallbacks(button, action, mods, x, y);
  });
  glfwSetScrollCallback(native, [](GLFWwindow *native, double x, double y) {
    auto window = FindWindow(native);
    if (!window)
      return;
    if (auto context = window->GetImGuiContext()) {
      ImGuiContextScope scope(context);
      ImGui_ImplGlfw_ScrollCallback(native, x, y);
    }
    window->ScrollEvent().InvokeCallbacks(x, y);
  });
  glfwSetKeyCallback(native, [](GLFWwindow *native, int key, int scancode, int action, int mods) {
    auto window = FindWindow(native);
    if (!window)
      return;
    if (auto context = window->GetImGuiContext()) {
      ImGuiContextScope scope(context);
      ImGui_ImplGlfw_KeyCallback(native, key, scancode, action, mods);
    }
    window->KeyEvent().InvokeCallbacks(key, scancode, action, mods);
  });
  glfwSetCharCallback(native, [](GLFWwindow *native, unsigned int codepoint) {
    auto window = FindWindow(native);
    if (!window)
      return;
    if (auto context = window->GetImGuiContext()) {
      ImGuiContextScope scope(context);
      ImGui_ImplGlfw_CharCallback(native, codepoint);
    }
    window->CharEvent().InvokeCallbacks(codepoint);
  });
  glfwSetDropCallback(native, [](GLFWwindow *native, int count, const char **paths) {
    auto window = FindWindow(native);
    if (!window)
      return;
    window->DropEvent().InvokeCallbacks(count, paths);
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
  std::vector<GLFWwindow *> destroyed;
  {
    std::lock_guard<std::mutex> lock(Get().mutex);
    destroyed.swap(Get().destroyed);
  }
  for (auto window : destroyed)
    glfwDestroyWindow(window);
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

void DestroyWindow(GLFWwindow *window) {
  if (!window)
    return;
  if (IsWindowThread())
    glfwDestroyWindow(window);
  else {
    std::lock_guard<std::mutex> lock(Get().mutex);
    Get().destroyed.push_back(window);
  }
}
}  // namespace glfw

}  // namespace grassland::rhi::backend
