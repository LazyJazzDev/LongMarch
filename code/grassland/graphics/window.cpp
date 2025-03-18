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

Window::Window(int width, int height, const std::string &title, bool fullscreen, bool resizable, bool enable_hdr)
    : enable_hdr_(enable_hdr) {
  InitializeGLFW();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  if (fullscreen) {
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);
    glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  } else {
    if (!resizable) {
      glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    } else {
      glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    }
  }

  window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);

  if (!window_) {
    throw std::runtime_error("Failed to create GLFW window");
  }

  glfwSetWindowUserPointer(window_, this);
  glfwSetWindowSizeCallback(window_, [](GLFWwindow *window, int width, int height) {
    Window *p_window = static_cast<Window *>(glfwGetWindowUserPointer(window));
    p_window->resize_event_.InvokeCallbacks(width, height);
  });
  glfwSetMouseButtonCallback(window_, [](GLFWwindow *window, int button, int action, int mods) {
    Window *p_window = static_cast<Window *>(glfwGetWindowUserPointer(window));
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    p_window->mouse_button_event_.InvokeCallbacks(button, action, mods, x, y);
  });
  glfwSetCursorPosCallback(window_, [](GLFWwindow *window, double x, double y) {
    Window *p_window = static_cast<Window *>(glfwGetWindowUserPointer(window));
    p_window->mouse_move_event_.InvokeCallbacks(x, y);
  });
  glfwSetScrollCallback(window_, [](GLFWwindow *window, double xoffset, double yoffset) {
    Window *p_window = static_cast<Window *>(glfwGetWindowUserPointer(window));
    p_window->scroll_event_.InvokeCallbacks(xoffset, yoffset);
  });
  glfwSetKeyCallback(window_, [](GLFWwindow *window, int key, int scancode, int action, int mods) {
    Window *p_window = static_cast<Window *>(glfwGetWindowUserPointer(window));
    p_window->key_event_.InvokeCallbacks(key, scancode, action, mods);
  });
  glfwSetCharCallback(window_, [](GLFWwindow *window, unsigned int codepoint) {
    Window *p_window = static_cast<Window *>(glfwGetWindowUserPointer(window));
    p_window->char_event_.InvokeCallbacks(codepoint);
  });
  glfwSetDropCallback(window_, [](GLFWwindow *window, int count, const char **paths) {
    Window *p_window = static_cast<Window *>(glfwGetWindowUserPointer(window));
    p_window->drop_event_.InvokeCallbacks(count, paths);
  });
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

std::string Window::GetTitle() const {
  return glfwGetWindowTitle(window_);
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

void Window::SetHDR(bool enable_hdr) {
  enable_hdr_ = enable_hdr;
  resize_event_.InvokeCallbacks(GetWidth(), GetHeight());
}

void Window::PyBind(pybind11::module &m) {
  pybind11::class_<Window, std::shared_ptr<Window>> window(m, "Window");
  window.def("__repr__",
             [](const Window &w) { return pybind11::str("<Window {}x{}>").format(w.GetWidth(), w.GetHeight()); });
  window.def("width", &Window::GetWidth);
  window.def("height", &Window::GetHeight);
  window.def("should_close", &Window::ShouldClose);
  window.def("resize", &Window::Resize, pybind11::arg("width"), pybind11::arg("height"));
  window.def("close", &Window::CloseWindow);
  window.def("set_title", &Window::SetTitle, pybind11::arg("title"));
  window.def("get_title", &Window::GetTitle);
  window.def("set_hdr", &Window::SetHDR, pybind11::arg("hdr"));
  window.def(
      "get_key", [](const Window &w, int key) { return glfwGetKey(w.GLFWWindow(), key); }, pybind11::arg("key"));
  window.def(
      "get_mouse_button", [](const Window &w, int button) { return glfwGetMouseButton(w.GLFWWindow(), button); },
      pybind11::arg("button"));
  window.def("get_cursor_pos", [](const Window &w) {
    std::pair<double, double> pos;
    glfwGetCursorPos(w.GLFWWindow(), &pos.first, &pos.second);
    return pos;
  });

  m.def("glfw_poll_events", []() { glfwPollEvents(); });
}

}  // namespace grassland::graphics
