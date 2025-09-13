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

void Window::PybindClassRegistration(py::classh<Window> &c) {
  c.def("__repr__", [](Window *window) {
    return py::str("Window(width={}, height={}, title='{}', hdr={})")
        .format(window->GetWidth(), window->GetHeight(), window->GetTitle(), window->enable_hdr_);
  });
  c.def("get_width", &Window::GetWidth, "Get the window width");
  c.def("get_height", &Window::GetHeight, "Get the window height");
  c.def("should_close", &Window::ShouldClose, "Check if the window should close");
  c.def("get_title", &Window::GetTitle, "Get the window title");
  c.def("set_title", &Window::SetTitle, "Set the window title");
  c.def("resize", &Window::Resize, py::arg("new_width"), py::arg("new_height"), "Resize the window");
  c.def("set_hdr", &Window::SetHDR, py::arg("enable_hdr"), "Enable or disable HDR rendering");
  c.def("init_imgui", &Window::InitImGui, py::arg("font_file_path") = nullptr, py::arg("font_size") = 13.0f,
        "Initialize ImGui for the window");
  c.def("terminate_imgui", &Window::TerminateImGui, "Terminate ImGui for the window");
  c.def("begin_imgui_frame", &Window::BeginImGuiFrame, "Begin a new ImGui frame");
  c.def("end_imgui_frame", &Window::EndImGuiFrame, "End the current ImGui frame");
  c.def(
      "register_resize_event",
      [](Window *window, py::function callback) {
        return window->ResizeEvent().RegisterCallback([callback](int width, int height) { callback(width, height); });
      },
      py::arg("callback"), "Add a callback for window resize event");
  c.def(
      "register_mouse_move_event",
      [](Window *window, py::function callback) {
        return window->MouseMoveEvent().RegisterCallback([callback](double x, double y) { callback(x, y); });
      },
      py::arg("callback"), "Add a callback for mouse move event");
  c.def(
      "register_mouse_button_event",
      [](Window *window, py::function callback) {
        return window->MouseButtonEvent().RegisterCallback(
            [callback](int button, int action, int mods, double x, double y) { callback(button, action, mods, x, y); });
      },
      py::arg("callback"), "Add a callback for mouse button event");
  c.def(
      "register_scroll_event",
      [](Window *window, py::function callback) {
        return window->ScrollEvent().RegisterCallback(
            [callback](double xoffset, double yoffset) { callback(xoffset, yoffset); });
      },
      py::arg("callback"), "Add a callback for scroll event");
  c.def(
      "register_key_event",
      [](Window *window, py::function callback) {
        return window->KeyEvent().RegisterCallback(
            [callback](int key, int scancode, int action, int mods) { callback(key, scancode, action, mods); });
      },
      py::arg("callback"), "Add a callback for key event");
  c.def(
      "register_char_event",
      [](Window *window, py::function callback) {
        return window->CharEvent().RegisterCallback([callback](uint32_t codepoint) { callback(codepoint); });
      },
      py::arg("callback"), "Add a callback for char event");
  c.def(
      "register_drop_event",
      [](Window *window, py::function callback) {
        return window->DropEvent().RegisterCallback([callback](int count, const char **paths) {
          std::vector<std::string> path_list;
          for (int i = 0; i < count; i++) {
            path_list.emplace_back(paths[i]);
          }
          callback(path_list);
        });
      },
      py::arg("callback"), "Add a callback for drop event");

  c.def_static("poll_events", &glfwPollEvents);
}

}  // namespace grassland::graphics
