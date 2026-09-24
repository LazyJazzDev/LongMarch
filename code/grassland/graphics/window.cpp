#include "grassland/graphics/window.h"

#if defined(__APPLE__) && !defined(LONGMARCH_HEADLESS)
#include "grassland/graphics/window_gestures.h"
#endif

namespace grassland::graphics {

#ifndef LONGMARCH_HEADLESS
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

#if defined(__APPLE__) && !defined(LONGMARCH_HEADLESS)
  magnify_monitor_ = detail::InstallMagnifyEvents(this);
#endif
  glfwSetWindowUserPointer(window_, this);
  glfwSetFramebufferSizeCallback(window_, [](GLFWwindow *window, int width, int height) {
    auto *owner = static_cast<Window *>(glfwGetWindowUserPointer(window));
    owner->framebuffer_resize_event_.InvokeCallbacks(width, height);
  });
  glfwSetCursorEnterCallback(window_, [](GLFWwindow *window, int entered) {
    auto *owner = static_cast<Window *>(glfwGetWindowUserPointer(window));
    owner->cursor_enter_event_.InvokeCallbacks(entered == GLFW_TRUE);
  });
  glfwSetWindowFocusCallback(window_, [](GLFWwindow *window, int focused) {
    auto *owner = static_cast<Window *>(glfwGetWindowUserPointer(window));
    owner->focus_event_.InvokeCallbacks(focused == GLFW_TRUE);
  });
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

#endif

Window::Window(Hosted, int width, int height, const std::string &title)
    : hosted_(true),
      hosted_size_(width, height),
      hosted_framebuffer_(width, height),
      hosted_title_(title),
      enable_hdr_(false) {
}

void Window::UpdateHostedSize(glm::ivec2 logical, glm::ivec2 framebuffer) {
  if (!hosted_)
    throw std::logic_error("Expected hosted window");
  if (logical != hosted_size_) {
    hosted_size_ = logical;
    resize_event_.InvokeCallbacks(logical.x, logical.y);
  }
  if (framebuffer != hosted_framebuffer_) {
    hosted_framebuffer_ = framebuffer;
    framebuffer_resize_event_.InvokeCallbacks(framebuffer.x, framebuffer.y);
  }
}

void Window::SendPointer(double x, double y) {
  hosted_cursor_ = {x, y};
  mouse_move_event_.InvokeCallbacks(x, y);
}

void Window::SendMouseButton(int button, int action, int mods) {
  if (action)
    hosted_buttons_.insert(button);
  else
    hosted_buttons_.erase(button);
  mouse_button_event_.InvokeCallbacks(button, action, mods, hosted_cursor_.x, hosted_cursor_.y);
}

void Window::SendKey(int key, int action, int mods) {
  if (action)
    hosted_keys_.insert(key);
  else
    hosted_keys_.erase(key);
  key_event_.InvokeCallbacks(key, 0, action, mods);
}

void Window::SendFocus(bool focused) {
  hosted_focused_ = focused;
  if (!focused) {
    hosted_keys_.clear();
    hosted_buttons_.clear();
  }
  focus_event_.InvokeCallbacks(focused);
}

Window::~Window() {
  CloseWindow();
}

int Window::GetWidth() const {
  if (hosted_) {
    return hosted_size_.x;
  }
#ifndef LONGMARCH_HEADLESS

  int width, height;
  glfwGetWindowSize(window_, &width, &height);
  return width;

#else
  return hosted_size_.x;
#endif
}

int Window::GetHeight() const {
  if (hosted_) {
    return hosted_size_.y;
  }
#ifndef LONGMARCH_HEADLESS

  int width, height;
  glfwGetWindowSize(window_, &width, &height);
  return height;

#else
  return hosted_size_.y;
#endif
}

glm::ivec2 Window::GetSize() const {
  if (hosted_) {
    return hosted_size_;
  }
#ifndef LONGMARCH_HEADLESS

  glm::ivec2 size;
  glfwGetWindowSize(window_, &size.x, &size.y);
  return size;

#else
  return hosted_size_;
#endif
}

glm::ivec2 Window::GetFramebufferSize() const {
  if (hosted_) {
    return hosted_framebuffer_;
  }
#ifndef LONGMARCH_HEADLESS

  glm::ivec2 size;
  glfwGetFramebufferSize(window_, &size.x, &size.y);
  return size;

#else
  return hosted_framebuffer_;
#endif
}

glm::dvec2 Window::GetCursorPosition() const {
  if (hosted_) {
    return hosted_cursor_;
  }
#ifndef LONGMARCH_HEADLESS

  glm::dvec2 position;
  glfwGetCursorPos(window_, &position.x, &position.y);
  return position;

#else
  return hosted_cursor_;
#endif
}

bool Window::IsKeyDown(int key) const {
  if (hosted_) {
    return hosted_keys_.count(key);
  }
#ifndef LONGMARCH_HEADLESS

  return glfwGetKey(window_, key) == GLFW_PRESS;

#else
  return hosted_keys_.count(key);
#endif
}

bool Window::IsMouseButtonDown(int button) const {
  if (hosted_) {
    return hosted_buttons_.count(button);
  }
#ifndef LONGMARCH_HEADLESS

  return glfwGetMouseButton(window_, button) == GLFW_PRESS;

#else
  return hosted_buttons_.count(button);
#endif
}

bool Window::IsFocused() const {
  if (hosted_) {
    return hosted_focused_;
  }
#ifndef LONGMARCH_HEADLESS

  return glfwGetWindowAttrib(window_, GLFW_FOCUSED) == GLFW_TRUE;

#else
  return hosted_focused_;
#endif
}

void Window::Focus() {
  if (hosted_) {
    SendFocus(true);
    return;
  }
#ifndef LONGMARCH_HEADLESS

  glfwFocusWindow(window_);

#else
  SendFocus(true);
  return;
#endif
}

void Window::RequestClose() {
  if (hosted_) {
    hosted_closed_ = true;
    return;
  }
#ifndef LONGMARCH_HEADLESS

  glfwSetWindowShouldClose(window_, GLFW_TRUE);

#else
  hosted_closed_ = true;
  return;
#endif
}

glm::ivec2 Window::GetPosition() const {
  if (hosted_) {
    return {0, 0};
  }
#ifndef LONGMARCH_HEADLESS

  glm::ivec2 position;
  glfwGetWindowPos(window_, &position.x, &position.y);
  return position;

#else
  return {0, 0};
#endif
}

void Window::SetPosition(int x, int y) {
  if (hosted_) {
    return;
  }
#ifndef LONGMARCH_HEADLESS

  glfwSetWindowPos(window_, x, y);

#else
  return;
#endif
}

glm::ivec4 Window::GetFrameSize() const {
  if (hosted_) {
    return {0, 0, 0, 0};
  }
#ifndef LONGMARCH_HEADLESS

  glm::ivec4 frame;
  glfwGetWindowFrameSize(window_, &frame.x, &frame.y, &frame.z, &frame.w);
  return frame;

#else
  return {0, 0, 0, 0};
#endif
}

glm::ivec4 Window::GetMonitorWorkArea() const {
  if (hosted_) {
    return {0, 0, hosted_size_.x, hosted_size_.y};
  }
#ifndef LONGMARCH_HEADLESS

  const auto position = GetPosition();
  const auto size = GetSize();
  int count = 0;
  auto **monitors = glfwGetMonitors(&count);
  auto *monitor = glfwGetPrimaryMonitor();
  int64_t best_overlap = 0;
  for (int i = 0; i < count; ++i) {
    glm::ivec4 area;
    glfwGetMonitorWorkarea(monitors[i], &area.x, &area.y, &area.z, &area.w);
    const auto overlap =
        glm::max(glm::ivec2{0}, glm::min(position + size, glm::ivec2{area.x + area.z, area.y + area.w}) -
                                    glm::max(position, glm::ivec2{area.x, area.y}));
    const int64_t pixels = int64_t(overlap.x) * overlap.y;
    if (pixels > best_overlap) {
      best_overlap = pixels;
      monitor = monitors[i];
    }
  }
  if (!monitor)
    return {position.x, position.y, size.x, size.y};
  glm::ivec4 area;
  glfwGetMonitorWorkarea(monitor, &area.x, &area.y, &area.z, &area.w);
  return area;

#else
  return {0, 0, hosted_size_.x, hosted_size_.y};
#endif
}

void Window::PollEvents() {
#ifndef LONGMARCH_HEADLESS

  glfwPollEvents();

#else
  return;
#endif
}

void Window::SetTitle(const std::string &title) {
  if (hosted_) {
    hosted_title_ = title;
    return;
  }
#ifndef LONGMARCH_HEADLESS

  glfwSetWindowTitle(window_, title.c_str());

#else
  hosted_title_ = title;
  return;
#endif
}

std::string Window::GetTitle() const {
  if (hosted_) {
    return hosted_title_;
  }
#ifndef LONGMARCH_HEADLESS

  return glfwGetWindowTitle(window_);

#else
  return hosted_title_;
#endif
}

void Window::Resize(int new_width, int new_height) {
  if (hosted_) {
    UpdateHostedSize({new_width, new_height}, {new_width, new_height});
    return;
  }
#ifndef LONGMARCH_HEADLESS

  glfwSetWindowSize(window_, new_width, new_height);

#else
  UpdateHostedSize({new_width, new_height}, {new_width, new_height});
  return;
#endif
}

void Window::CloseWindow() {
  if (hosted_) {
    hosted_closed_ = true;
    return;
  }
#ifndef LONGMARCH_HEADLESS

#if defined(__APPLE__) && !defined(LONGMARCH_HEADLESS)
  detail::RemoveMagnifyEvents(magnify_monitor_);
  magnify_monitor_ = nullptr;
#endif
  glfwDestroyWindow(window_);
  window_ = nullptr;

#else
  hosted_closed_ = true;
  return;
#endif
}

bool Window::ShouldClose() const {
  if (hosted_) {
    return hosted_closed_;
  }
#ifndef LONGMARCH_HEADLESS

  return glfwWindowShouldClose(window_);

#else
  return hosted_closed_;
#endif
}

void Window::SetHDR(bool enable_hdr) {
  enable_hdr_ = enable_hdr;
  resize_event_.InvokeCallbacks(GetWidth(), GetHeight());
}

#if defined(LONGMARCH_PYTHON_ENABLED)
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

  c.def("get_size", [](Window &w) {
    auto v = w.GetSize();
    return py::make_tuple(v.x, v.y);
  });
  c.def("get_framebuffer_size", [](Window &w) {
    auto v = w.GetFramebufferSize();
    return py::make_tuple(v.x, v.y);
  });
  c.def("get_cursor_position", [](Window &w) {
    auto v = w.GetCursorPosition();
    return py::make_tuple(v.x, v.y);
  });
  c.def("get_position", [](Window &w) {
    auto v = w.GetPosition();
    return py::make_tuple(v.x, v.y);
  });
  c.def("set_position", &Window::SetPosition, py::arg("x"), py::arg("y"));
  c.def("get_frame_size", [](Window &w) {
    auto v = w.GetFrameSize();
    return py::make_tuple(v.x, v.y, v.z, v.w);
  });
  c.def("get_monitor_work_area", [](Window &w) {
    auto v = w.GetMonitorWorkArea();
    return py::make_tuple(v.x, v.y, v.z, v.w);
  });
  c.def("is_key_down", &Window::IsKeyDown, py::arg("key"));
  c.def("is_mouse_button_down", &Window::IsMouseButtonDown, py::arg("button"));
  c.def("is_focused", &Window::IsFocused);
  c.def("focus", &Window::Focus);
  c.def("request_close", &Window::RequestClose);
  c.def(
      "register_framebuffer_resize_event",
      [](Window &w, py::function callback) {
        return w.FramebufferResizeEvent().RegisterCallback(
            [callback](int width, int height) { callback(width, height); });
      },
      py::arg("callback"));
  c.def("unregister_framebuffer_resize_event",
        [](Window &w, uint32_t id) { w.FramebufferResizeEvent().UnregisterCallback(id); });
  c.def(
      "register_cursor_enter_event",
      [](Window &w, py::function callback) {
        return w.CursorEnterEvent().RegisterCallback([callback](bool entered) { callback(entered); });
      },
      py::arg("callback"));
  c.def("unregister_cursor_enter_event", [](Window &w, uint32_t id) { w.CursorEnterEvent().UnregisterCallback(id); });
  c.def(
      "register_focus_event",
      [](Window &w, py::function callback) {
        return w.FocusEvent().RegisterCallback([callback](bool focused) { callback(focused); });
      },
      py::arg("callback"));
  c.def("unregister_focus_event", [](Window &w, uint32_t id) { w.FocusEvent().UnregisterCallback(id); });
  c.def_static("poll_events", &Window::PollEvents);
}
#endif

}  // namespace grassland::graphics
