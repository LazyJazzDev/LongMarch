#include <long_march.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

using namespace long_march;

namespace {
const char *PipelineName(sparkium::RenderPipeline pipeline) {
  switch (pipeline) {
    case sparkium::RENDER_PIPELINE_RT_FALLBACK:
      return "Compute ray tracing";
    case sparkium::RENDER_PIPELINE_RASTERIZATION: return "Rasterization";
    case sparkium::RENDER_PIPELINE_RAY_TRACING: return "Ray tracing";
    default: return "Auto";
  }
}

void ResizeWindowForFilm(graphics::Window *window, sparkium::Film *film) {
  int window_x = 0;
  int window_y = 0;
  int window_width = window->GetWidth();
  int window_height = window->GetHeight();
  glfwGetWindowPos(window->GLFWWindow(), &window_x, &window_y);

  int monitor_count = 0;
  GLFWmonitor **monitors = glfwGetMonitors(&monitor_count);
  GLFWmonitor *monitor = glfwGetPrimaryMonitor();
  int best_overlap = -1;
  for (int i = 0; i < monitor_count; ++i) {
    int monitor_x, monitor_y, monitor_width, monitor_height;
    glfwGetMonitorWorkarea(monitors[i], &monitor_x, &monitor_y, &monitor_width, &monitor_height);
    const int overlap_width =
        std::max(0, std::min(window_x + window_width, monitor_x + monitor_width) - std::max(window_x, monitor_x));
    const int overlap_height =
        std::max(0, std::min(window_y + window_height, monitor_y + monitor_height) - std::max(window_y, monitor_y));
    const int overlap = overlap_width * overlap_height;
    if (overlap > best_overlap) {
      best_overlap = overlap;
      monitor = monitors[i];
    }
  }

  int work_x, work_y, work_width, work_height;
  glfwGetMonitorWorkarea(monitor, &work_x, &work_y, &work_width, &work_height);
  int frame_left, frame_top, frame_right, frame_bottom;
  glfwGetWindowFrameSize(window->GLFWWindow(), &frame_left, &frame_top, &frame_right, &frame_bottom);
  const int available_width = std::max(1, work_width - frame_left - frame_right);
  const int available_height = std::max(1, work_height - frame_top - frame_bottom);
  const float scale = std::min({1.0f, static_cast<float>(available_width) / film->GetWidth(),
                                static_cast<float>(available_height) / film->GetHeight()});
  const int target_width = std::max(1, static_cast<int>(std::floor(film->GetWidth() * scale)));
  const int target_height = std::max(1, static_cast<int>(std::floor(film->GetHeight() * scale)));

  window->Resize(target_width, target_height);
  glfwSetWindowPos(window->GLFWWindow(), work_x + frame_left + (available_width - target_width) / 2,
                   work_y + frame_top + (available_height - target_height) / 2);
}
}  // namespace

int main(int argc, char **argv) {
  try {
    std::filesystem::path input = argc > 1 ? argv[1] : std::filesystem::path(FindAssetPath("scenes"));
    std::vector<std::filesystem::path> scene_files;
    if (std::filesystem::is_regular_file(input)) scene_files.push_back(std::filesystem::absolute(input));
    else scene_files = sparkium::FindJsonScenes(input);
    if (scene_files.empty()) throw std::runtime_error("no scene.json files found under: " + input.string());

    std::unique_ptr<graphics::Core> graphics_core;
    if (graphics::CreateCore(graphics::BACKEND_API_DEFAULT, graphics::Core::Settings{}, &graphics_core) != 0)
      throw std::runtime_error("failed to create graphics core");
    graphics_core->InitializeLogicalDeviceAutoSelect(false);
    sparkium::Core core(graphics_core.get());

    std::unique_ptr<sparkium::JsonScene> loaded;
    std::unique_ptr<graphics::Image> image;
    std::unique_ptr<graphics::Window> window;
    size_t selected = 0;
    bool resize_pending = true;
    sparkium::RenderPipeline pipeline = sparkium::RENDER_PIPELINE_AUTO;
    std::string load_error;
    auto load_selected = [&]() {
      load_error.clear();
      auto next = sparkium::JsonScene::Load(&core, scene_files[selected], &load_error);
      if (!next) return false;
      loaded = std::move(next);
      pipeline = loaded->GetRenderPipeline();
      auto *film = loaded->GetFilm();
      graphics_core->CreateImage(film->GetWidth(), film->GetHeight(), graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image);
      resize_pending = true;
      return true;
    };
    if (!load_selected()) throw std::runtime_error(load_error);

    graphics_core->CreateWindowObject(loaded->GetFilm()->GetWidth(), loaded->GetFilm()->GetHeight(),
                                      "Sparkium Scene Browser", false, true, &window);
    ResizeWindowForFilm(window.get(), loaded->GetFilm());
    resize_pending = false;
    window->InitImGui(nullptr, 18.0f);
    FPSCounter fps_counter;
    bool show_browser = true;

    while (!window->ShouldClose()) {
      window->BeginImGuiFrame();
      ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
      ImGui::SetNextWindowBgAlpha(0.85f);
      ImGui::Begin("Sparkium scenes", &show_browser, ImGuiWindowFlags_AlwaysAutoResize);
      if (ImGui::BeginCombo("Scene", loaded->GetName().c_str())) {
        for (size_t i = 0; i < scene_files.size(); ++i) {
          bool current = i == selected;
          if (ImGui::Selectable(scene_files[i].parent_path().filename().string().c_str(), current)) {
            auto previous = selected;
            selected = i;
            if (!load_selected()) selected = previous;
          }
          if (current) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
      }
      int pipeline_index = static_cast<int>(pipeline);
      const char *pipelines[] = {"Rasterization", "Ray tracing", "Auto", "Compute ray tracing"};
      if (ImGui::Combo("Pipeline", &pipeline_index, pipelines, 4)) {
        pipeline = static_cast<sparkium::RenderPipeline>(pipeline_index);
        loaded->GetFilm()->Reset();
      }
      int &samples = loaded->GetScene()->settings.samples_per_dispatch;
      if (ImGui::SliderInt("Samples / frame", &samples, 1, 256)) loaded->GetFilm()->Reset();
      if (ImGui::Button("Reload")) load_selected();
      ImGui::SameLine();
      if (ImGui::Button("Reset film")) loaded->GetFilm()->Reset();
      ImGui::Text("%s", scene_files[selected].string().c_str());
      ImGui::Text("Pipeline: %s", PipelineName(pipeline));
      if (!load_error.empty()) ImGui::TextColored({1, .3f, .3f, 1}, "%s", load_error.c_str());
      ImGui::Text("%.1f FPS", fps_counter.TickFPS());
      ImGui::End();
      window->EndImGuiFrame();

      core.Render(loaded->GetScene(), loaded->GetCamera(), loaded->GetFilm(), pipeline);
      loaded->GetFilm()->Develop(image.get());
      std::unique_ptr<graphics::CommandContext> command_context;
      graphics_core->CreateCommandContext(&command_context);
      command_context->CmdPresent(window.get(), image.get());
      graphics_core->SubmitCommandContext(command_context.get());
      glfwPollEvents();
      if (resize_pending) {
        ResizeWindowForFilm(window.get(), loaded->GetFilm());
        resize_pending = false;
      }
    }
    window->TerminateImGui();
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "sparkium_gui: " << exception.what() << '\n';
    return 1;
  }
}
