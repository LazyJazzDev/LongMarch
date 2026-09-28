#include "pybind/pybind.h"

namespace grassland::graphics::pybind {
namespace {
py::dict DisplayBrightnessDict(const DisplayBrightness &info) {
  py::dict result;
  result["sdr_white_nits"] = info.sdr_white_nits;
  result["hdr_reference_white_scale"] = info.hdr_reference_white_scale;
  result["hdr_headroom"] = info.hdr_headroom;
  result["reference_white_known"] = info.reference_white_known;
  result["hdr_enabled"] = info.hdr_enabled;
  return result;
}
}  // namespace

void RegisterWindow(py::classh<Window> &c) {
  c.def("set_hdr_brightness_alignment", &Window::SetHDRBrightnessAlignment);
  c.def("hdr_brightness_alignment", &Window::HDRBrightnessAlignment);
  c.def("hdr_reference_white_scale", &Window::HDRReferenceWhiteScale);
  c.def("refresh_display_brightness", &Window::RefreshDisplayBrightness);
  c.def("display_brightness", [](Window &window) { return DisplayBrightnessDict(window.GetDisplayBrightness()); });
  c.def(
      "register_display_brightness_event",
      [](Window &w, py::function callback) {
        return w.DisplayBrightnessEvent().RegisterCallback(
            [callback](const DisplayBrightness &info) { callback(DisplayBrightnessDict(info)); });
      },
      py::arg("callback"));
  c.def("unregister_display_brightness_event",
        [](Window &w, uint32_t id) { w.DisplayBrightnessEvent().UnregisterCallback(id); });
  c.def("is_hdr", &Window::IsHDR);
  c.def("__repr__", [](Window *window) {
    return py::str("Window(width={}, height={}, title='{}', hdr={})")
        .format(window->GetWidth(), window->GetHeight(), window->GetTitle(), window->IsHDR());
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
  c.def("unregister_resize_event", [](Window &w, uint32_t id) { w.ResizeEvent().UnregisterCallback(id); });
  c.def("unregister_mouse_move_event", [](Window &w, uint32_t id) { w.MouseMoveEvent().UnregisterCallback(id); });
  c.def("unregister_mouse_button_event", [](Window &w, uint32_t id) { w.MouseButtonEvent().UnregisterCallback(id); });
  c.def("unregister_scroll_event", [](Window &w, uint32_t id) { w.ScrollEvent().UnregisterCallback(id); });
  c.def("unregister_key_event", [](Window &w, uint32_t id) { w.KeyEvent().UnregisterCallback(id); });
  c.def("unregister_char_event", [](Window &w, uint32_t id) { w.CharEvent().UnregisterCallback(id); });
  c.def("unregister_drop_event", [](Window &w, uint32_t id) { w.DropEvent().UnregisterCallback(id); });
  c.def("supports_magnify_gestures", &Window::SupportsMagnifyGestures);
  c.def(
      "register_magnify_event",
      [](Window &w, py::function callback) {
        return w.MagnifyEvent().RegisterCallback([callback](const MagnifyGesture &gesture) {
          callback(gesture.scale, gesture.x, gesture.y, gesture.phase);
        });
      },
      py::arg("callback"), "Add a callback(scale, x, y, phase) for native magnify gestures");
  c.def("unregister_magnify_event", [](Window &w, uint32_t id) { w.MagnifyEvent().UnregisterCallback(id); });
  c.def_static("poll_events", &Window::PollEvents);
}
}  // namespace grassland::graphics::pybind
