#include <long_march.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

#include "../sparkium_backend.h"

using namespace long_march;

namespace {
const char *PipelineName(sparkium::RenderPipeline pipeline) {
  switch (pipeline) {
    case sparkium::RENDER_PIPELINE_RAY_QUERY:
      return "Path Tracing - Ray Query";
    case sparkium::RENDER_PIPELINE_RT_FALLBACK:
      return "Path Tracing - Fallback";
    case sparkium::RENDER_PIPELINE_RASTERIZATION:
      return "Rasterization";
    case sparkium::RENDER_PIPELINE_RAY_TRACING:
      return "Path Tracing";
    default:
      return "Auto";
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
    std::filesystem::path input = FindAssetPath("scenes");
    bool input_selected = false;
    bool hdr_requested = false;
    auto backend = graphics::BACKEND_API_DEFAULT;
    int frame_limit = 0;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--help") {
        std::cout
            << "Usage: sparkium_gui [scene.json|directory] [--backend auto|metal|vulkan|d3d12] [--hdr] [--frames N]\n";
        return 0;
      }
      if (arg == "--hdr")
        hdr_requested = true;
      else if (arg == "--backend" && i + 1 < argc)
        backend = ParseSparkiumBackend(argv[++i]);
      else if (arg == "--frames" && i + 1 < argc) {
        frame_limit = std::stoi(argv[++i]);
        if (frame_limit <= 0)
          throw std::invalid_argument("--frames must be positive");
      } else if (!arg.empty() && arg[0] != '-' && !input_selected) {
        input = arg;
        input_selected = true;
      } else
        throw std::invalid_argument("unknown or incomplete argument: " + arg);
    }
    std::vector<std::filesystem::path> scene_files;
    if (std::filesystem::is_regular_file(input))
      scene_files.push_back(std::filesystem::absolute(input));
    else
      scene_files = sparkium::FindJsonScenes(input);
    if (scene_files.empty())
      throw std::runtime_error("no scene.json files found under: " + input.string());

    std::unique_ptr<graphics::Core> graphics_core;
    if (graphics::CreateCore(backend, graphics::Core::Settings{}, &graphics_core) != 0)
      throw std::runtime_error("failed to create graphics core");
    if (graphics_core->InitializeLogicalDeviceAutoSelect(false) != 0)
      throw std::runtime_error("failed to initialize graphics device");
    const bool hdr_available = graphics_core->API() == graphics::BACKEND_API_METAL;
    if (hdr_requested && !hdr_available)
      throw std::invalid_argument("HDR preview currently requires the Metal backend");
    bool hdr_active = false;
    sparkium::Core core(graphics_core.get());

    std::unique_ptr<sparkium::JsonScene> loaded;
    std::unique_ptr<graphics::Image> image;
    std::unique_ptr<graphics::Window> window;
    size_t selected = 0;
    bool resize_pending = true;
    sparkium::RenderPipeline pipeline = sparkium::RENDER_PIPELINE_AUTO;
    std::string load_error;
    auto create_display_image = [&]() {
      auto *film = loaded->GetFilm();
      graphics_core->WaitGPU();
      graphics_core->CreateImage(
          film->GetWidth(), film->GetHeight(),
          hdr_active ? graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT : graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image);
    };
    auto load_selected = [&]() {
      load_error.clear();
      auto next = sparkium::JsonScene::Load(&core, scene_files[selected], &load_error);
      if (!next)
        return false;
      loaded = std::move(next);
      pipeline = loaded->GetRenderPipeline();
      create_display_image();
      resize_pending = true;
      return true;
    };
    if (!load_selected())
      throw std::runtime_error(load_error);

    graphics_core->CreateWindowObject(loaded->GetFilm()->GetWidth(), loaded->GetFilm()->GetHeight(),
                                      "Sparkium Scene Browser", false, true, &window);
    ResizeWindowForFilm(window.get(), loaded->GetFilm());
    resize_pending = false;
    window->InitImGui(nullptr, 18.0f);
    FPSCounter fps_counter;
    bool show_browser = true;

    int rendered_frames = 0;
    while (!window->ShouldClose() && (!frame_limit || rendered_frames++ < frame_limit)) {
      // Apply before BeginImGuiFrame so ImGui and presentation use the same format.
      if (hdr_requested != hdr_active) {
        window->SetHDR(hdr_requested);
        hdr_active = hdr_requested;
        create_display_image();
      }
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
            if (!load_selected())
              selected = previous;
          }
          if (current)
            ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
      }
      const std::string auto_label =
          std::string("Auto (") + PipelineName(core.ResolveRenderPipeline(sparkium::RENDER_PIPELINE_AUTO)) + ")";
      const auto selected_pipeline =
          pipeline == sparkium::RENDER_PIPELINE_AUTO ? pipeline : core.ResolveRenderPipeline(pipeline);
      const char *pipeline_label =
          selected_pipeline == sparkium::RENDER_PIPELINE_AUTO ? auto_label.c_str() : PipelineName(selected_pipeline);
      if (ImGui::BeginCombo("Pipeline", pipeline_label)) {
        for (auto option : {sparkium::RENDER_PIPELINE_AUTO, sparkium::RENDER_PIPELINE_RASTERIZATION,
                            sparkium::RENDER_PIPELINE_RAY_TRACING, sparkium::RENDER_PIPELINE_RT_FALLBACK,
                            sparkium::RENDER_PIPELINE_RAY_QUERY}) {
          if (option == sparkium::RENDER_PIPELINE_RAY_TRACING && !graphics_core->DeviceRayTracingSupport())
            continue;
          if (option == sparkium::RENDER_PIPELINE_RAY_QUERY && !graphics_core->DeviceRayQuerySupport())
            continue;
          const bool current = option == selected_pipeline;
          const char *label = option == sparkium::RENDER_PIPELINE_AUTO ? auto_label.c_str() : PipelineName(option);
          if (ImGui::Selectable(label, current) && pipeline != option) {
            pipeline = option;
            loaded->GetFilm()->Reset();
          }
          if (current)
            ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
      }
      ImGui::BeginDisabled(!hdr_available);
      ImGui::Checkbox("HDR preview", &hdr_requested);
      ImGui::EndDisabled();
      if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip(
            hdr_available
                ? "Linear HDR with exposure; bypasses SDR view transform, gamma and contrast. Requires an HDR display."
                : "HDR preview currently requires the Metal backend.");
      ImGui::SliderFloat("Exposure (EV)", &loaded->GetFilm()->info.exposure, -8.0f, 8.0f, "%.2f");
      ImGui::TextUnformatted(hdr_active ? "Display: HDR (linear)" : "Display: SDR (scene view transform)");
      int &samples = loaded->GetScene()->settings.samples_per_dispatch;
      if (ImGui::SliderInt("Samples / frame", &samples, 1, 256))
        loaded->GetFilm()->Reset();
      if (ImGui::Button("Reload"))
        load_selected();
      ImGui::SameLine();
      if (ImGui::Button("Reset film"))
        loaded->GetFilm()->Reset();
      ImGui::Text("%s", scene_files[selected].string().c_str());
      ImGui::Text("Backend: %s", graphics::BackendAPIString(graphics_core->API()));
      const auto resolved_pipeline = core.ResolveRenderPipeline(pipeline);
      if (resolved_pipeline != pipeline)
        ImGui::Text("Pipeline: %s (%s)", PipelineName(pipeline), PipelineName(resolved_pipeline));
      else
        ImGui::Text("Pipeline: %s", PipelineName(pipeline));
      const float fps = fps_counter.TickFPS();
      if (resolved_pipeline == sparkium::RENDER_PIPELINE_RASTERIZATION) {
        ImGui::TextUnformatted("Ray/s: N/A");
        ImGui::TextUnformatted("Accumulated spp: N/A");
      } else {
        const auto *film = loaded->GetFilm();
        const double camera_rays_per_second = film->info.accumulated_samples > 0
                                                  ? static_cast<double>(film->GetWidth()) * film->GetHeight() *
                                                        loaded->GetScene()->settings.samples_per_dispatch * fps
                                                  : 0.0;
        ImGui::Text("Ray/s: %.2f M", camera_rays_per_second / 1e6);
        if (ImGui::IsItemHovered())
          ImGui::SetTooltip("Camera rays: width x height x samples per frame x FPS.");
        ImGui::Text("Accumulated spp: %d", film->info.accumulated_samples);
      }
      if (!load_error.empty())
        ImGui::TextColored({1, .3f, .3f, 1}, "%s", load_error.c_str());
      ImGui::Text("%.1f FPS", fps);
      ImGui::End();
      window->EndImGuiFrame();

      core.Render(loaded->GetScene(), loaded->GetCamera(), loaded->GetFilm(), pipeline);
      loaded->GetFilm()->Develop(image.get(), hdr_active);
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
