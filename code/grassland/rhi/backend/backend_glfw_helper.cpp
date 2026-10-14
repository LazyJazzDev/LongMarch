#include "grassland/rhi/backend/backend_glfw_helper.h"

#include <mutex>
#include <thread>

#include "grassland/rhi/rhi_util.h"
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

  ~State() {
    if (initialized && thread == std::this_thread::get_id())
      glfwTerminate();
  }
};

State &Get() {
  static State state;
  return state;
}
}  // namespace

bool Initialize() {
  auto &state = Get();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.initialized)
    return state.thread == std::this_thread::get_id();
  if (!glfwInit()) {
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
  return true;
}

GLFWwindow *CreateWindow(int width, int height, const char *title) {
  if (width <= 0 || height <= 0) {
    ReportError(-1, "Window dimensions must be positive.");
    return nullptr;
  }
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
  if (!window || !IsWindowThread())
    return false;
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
