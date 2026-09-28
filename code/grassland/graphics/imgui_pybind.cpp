#include "grassland/graphics/imgui_pybind.h"

#include "grassland/graphics/window.h"

namespace grassland::graphics {
#if defined(LONGMARCH_PYTHON_ENABLED)
// A small immediate-mode subset for Python demos. Calls act on the current ImGui
// context, which Window.begin_imgui_frame selects; widgets belong between
// begin_imgui_frame and end_imgui_frame. Text is never used as a format string.
void PybindImGuiRegistration(py::module_ &m) {
  m.doc() = "Dear ImGui widgets for the window created by long_march.graphics";

  m.attr("COND_ALWAYS") = int(ImGuiCond_Always);
  m.attr("COND_ONCE") = int(ImGuiCond_Once);
  m.attr("COND_FIRST_USE_EVER") = int(ImGuiCond_FirstUseEver);
  m.attr("WINDOW_FLAGS_NONE") = int(ImGuiWindowFlags_None);
  m.attr("WINDOW_FLAGS_NO_MOVE") = int(ImGuiWindowFlags_NoMove);
  m.attr("WINDOW_FLAGS_NO_RESIZE") = int(ImGuiWindowFlags_NoResize);
  m.attr("WINDOW_FLAGS_ALWAYS_AUTO_RESIZE") = int(ImGuiWindowFlags_AlwaysAutoResize);
  m.attr("SLIDER_FLAGS_NONE") = int(ImGuiSliderFlags_None);
  m.attr("SLIDER_FLAGS_LOGARITHMIC") = int(ImGuiSliderFlags_Logarithmic);
  m.attr("SLIDER_FLAGS_ALWAYS_CLAMP") = int(ImGuiSliderFlags_AlwaysClamp);
  m.attr("TREE_NODE_FLAGS_DEFAULT_OPEN") = int(ImGuiTreeNodeFlags_DefaultOpen);

  m.def(
      "set_current_context", [](Window *window) { ImGui::SetCurrentContext(window->GetImGuiContext()); },
      py::arg("window"), "Select the ImGui context of a window initialized with init_imgui");
  m.def(
      "set_ini_filename",
      [](std::optional<std::string> filename) {
        // ImGui keeps the pointer, so the name must outlive the context.
        static std::string storage;
        storage = filename.value_or("");
        ImGui::GetIO().IniFilename = filename ? storage.c_str() : nullptr;
      },
      py::arg("filename"), "Set the layout file of the current context; None disables saving");
  m.def("want_capture_mouse", []() { return ImGui::GetIO().WantCaptureMouse; });
  m.def("want_capture_keyboard", []() { return ImGui::GetIO().WantCaptureKeyboard; });

  m.def(
      "set_next_window_pos", [](float x, float y, int cond) { ImGui::SetNextWindowPos({x, y}, cond); }, py::arg("x"),
      py::arg("y"), py::arg("cond") = int(ImGuiCond_None));
  m.def("set_next_window_bg_alpha", &ImGui::SetNextWindowBgAlpha, py::arg("alpha"));
  m.def(
      "begin", [](const std::string &name, int flags) { return ImGui::Begin(name.c_str(), nullptr, flags); },
      py::arg("name"), py::arg("flags") = 0, "Begin a window; always call end(), even when this returns False");
  m.def("end", []() { ImGui::End(); });

  m.def(
      "text", [](const std::string &text) { ImGui::TextUnformatted(text.c_str(), text.c_str() + text.size()); },
      py::arg("text"));
  m.def("separator", []() { ImGui::Separator(); });
  m.def("new_line", []() { ImGui::NewLine(); });
  m.def(
      "same_line", [](float offset, float spacing) { ImGui::SameLine(offset, spacing); }, py::arg("offset") = 0.0f,
      py::arg("spacing") = -1.0f);
  m.def(
      "collapsing_header",
      [](const std::string &label, int flags) { return ImGui::CollapsingHeader(label.c_str(), flags); },
      py::arg("label"), py::arg("flags") = 0);
  m.def("button", [](const std::string &label) { return ImGui::Button(label.c_str()); }, py::arg("label"));
  m.def(
      "checkbox",
      [](const std::string &label, bool value) {
        bool changed = ImGui::Checkbox(label.c_str(), &value);
        return py::make_tuple(changed, value);
      },
      py::arg("label"), py::arg("value"), "Return (changed, value)");
  m.def(
      "slider_float",
      [](const std::string &label, float value, float v_min, float v_max, const std::string &format, int flags) {
        bool changed = ImGui::SliderFloat(label.c_str(), &value, v_min, v_max, format.c_str(), flags);
        return py::make_tuple(changed, value);
      },
      py::arg("label"), py::arg("value"), py::arg("v_min"), py::arg("v_max"), py::arg("format") = "%.3f",
      py::arg("flags") = 0, "Return (changed, value)");
  m.def(
      "slider_int",
      [](const std::string &label, int value, int v_min, int v_max, const std::string &format, int flags) {
        bool changed = ImGui::SliderInt(label.c_str(), &value, v_min, v_max, format.c_str(), flags);
        return py::make_tuple(changed, value);
      },
      py::arg("label"), py::arg("value"), py::arg("v_min"), py::arg("v_max"), py::arg("format") = "%d",
      py::arg("flags") = 0, "Return (changed, value)");
  m.def(
      "plot_lines",
      [](const std::string &label, const std::vector<float> &values, float scale_min, float scale_max,
         float graph_width, float graph_height) {
        ImGui::PlotLines(label.c_str(), values.data(), int(values.size()), 0, nullptr, scale_min, scale_max,
                         {graph_width, graph_height});
      },
      py::arg("label"), py::arg("values"), py::arg("scale_min") = FLT_MAX, py::arg("scale_max") = FLT_MAX,
      py::arg("graph_width") = 0.0f, py::arg("graph_height") = 0.0f);
}
#endif
}  // namespace grassland::graphics
