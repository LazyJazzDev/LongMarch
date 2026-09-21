#include <long_march.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>

#include "../sparkium_backend.h"
#include "render_worker.h"

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

void ResizeWindowForFilm(graphics::Window *window, int film_width, int film_height) {
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
  const float scale = std::min(
      {1.0f, static_cast<float>(available_width) / film_width, static_cast<float>(available_height) / film_height});
  const int target_width = std::max(1, static_cast<int>(std::floor(film_width * scale)));
  const int target_height = std::max(1, static_cast<int>(std::floor(film_height * scale)));

  window->Resize(target_width, target_height);
  glfwSetWindowPos(window->GLFWWindow(), work_x + frame_left + (available_width - target_width) / 2,
                   work_y + frame_top + (available_height - target_height) / 2);
}
}  // namespace

int main(int argc, char **argv) {
  try {
    std::filesystem::path input = FindAssetPath("scenes");
    sparkium::BackendSelection backend;
    auto display_backend = graphics::BACKEND_API_DEFAULT;
    int frame_limit = 0;
    bool have_input = false;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--backend" && i + 1 < argc)
        backend = ParseSparkiumBackend(argv[++i]);
      else if (arg == "--display-backend" && i + 1 < argc)
        display_backend = sparkium::ToGraphicsBackend(ParseSparkiumBackend(argv[++i]));
      else if (arg == "--frames" && i + 1 < argc) {
        frame_limit = std::stoi(argv[++i]);
        if (frame_limit <= 0)
          throw std::invalid_argument("--frames must be positive");
      } else if (arg == "--help") {
        std::cout << "Usage: sparkium_gui [scene.json|directory] [--backend auto|cpu|cuda|d3d12|vulkan|metal] "
                     "[--display-backend auto|d3d12|vulkan|metal] [--frames N]\n";
        return 0;
      } else if (!have_input && arg.rfind("-", 0) != 0) {
        input = arg;
        have_input = true;
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

    // GLFW/ImGui and the display device belong exclusively to the main thread.
    std::unique_ptr<graphics::Core> display;
    if (graphics::CreateCore(display_backend, {}, &display) != 0 ||
        display->InitializeLogicalDeviceAutoSelect(false) != 0)
      throw std::runtime_error("failed to initialize display device");
    std::unique_ptr<graphics::Window> window;
    if (display->CreateWindowObject(960, 640, "Sparkium Scene Browser", false, true, &window) != 0)
      throw std::runtime_error("failed to create display window");
    window->InitImGui(nullptr, 18.0f);
    std::unique_ptr<graphics::Image> image;
    auto clear_preview = [&] {
      display->WaitGPU();
      if (!image && display->CreateImage(1, 1, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image) != 0)
        throw std::runtime_error("failed to create display image");
      std::unique_ptr<graphics::CommandContext> clear;
      display->CreateCommandContext(&clear);
      clear->CmdClearImage(image.get(), {24.0f / 255, 24.0f / 255, 24.0f / 255, 1.0f});
      display->SubmitCommandContext(clear.get());
    };
    clear_preview();
    sparkium_gui::RenderWorker worker(frame_limit);
    sparkium_gui::RenderRequest request;
    request.backend = backend.backend;
    request.graphics_api = backend.graphics_api;
    size_t selected = 0;
    request.scene = scene_files[selected];
    worker.Submit(request);
    std::shared_ptr<const sparkium_gui::RenderFrame> shown;
    FPSCounter fps_counter;
    int exit_code = 0;

    auto submit = [&] {
      worker.Submit(request);
      shown.reset();
      clear_preview();
    };
    while (!window->ShouldClose()) {
      const auto ui_start = std::chrono::steady_clock::now();
      glfwPollEvents();
      auto status = worker.Snapshot();
      if (status.frame && status.frame != shown) {
        const auto &frame = *status.frame;
        if (image->Extent().width != frame.width || image->Extent().height != frame.height) {
          display->WaitGPU();
          if (display->CreateImage(frame.width, frame.height, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image) != 0)
            throw std::runtime_error("failed to resize display image");
          ResizeWindowForFilm(window.get(), frame.width, frame.height);
        }
        image->UploadData(frame.rgba.data());
        shown = status.frame;
      }
      window->BeginImGuiFrame();
      ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
      ImGui::SetNextWindowBgAlpha(0.85f);
      ImGui::Begin("Sparkium scenes", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
      const auto scene_label = scene_files[selected].parent_path().filename().string();
      if (ImGui::BeginCombo("Scene", scene_label.c_str())) {
        for (size_t i = 0; i < scene_files.size(); ++i) {
          const bool current = i == selected;
          ImGui::PushID(static_cast<int>(i));
          if (ImGui::Selectable(scene_files[i].parent_path().filename().string().c_str(), current)) {
            selected = i;
            request.scene = scene_files[selected];
            request.pipeline.reset();
            request.samples.reset();
            submit();
          }
          if (current)
            ImGui::SetItemDefaultFocus();
          ImGui::PopID();
        }
        ImGui::EndCombo();
      }
      const char *backend_label = sparkium::BackendName(request.backend);
      if (ImGui::BeginCombo("Render backend", backend_label)) {
        for (auto option :
             {sparkium::RenderBackend::Graphics, sparkium::RenderBackend::CPU, sparkium::RenderBackend::CUDA}) {
          if (!sparkium::SupportBackend(option))
            continue;
          const bool current = option == request.backend;
          const char *label = sparkium::BackendName(option);
          if (ImGui::Selectable(label, current) && !current) {
            request.backend = option;
            request.pipeline = sparkium::RENDER_PIPELINE_AUTO;
            submit();
          }
          if (current)
            ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
      }
      if (request.backend == sparkium::RenderBackend::Graphics &&
          ImGui::BeginCombo("Graphics API", graphics::BackendAPIString(request.graphics_api))) {
        for (auto api : {graphics::BACKEND_API_D3D12, graphics::BACKEND_API_VULKAN, graphics::BACKEND_API_METAL}) {
          if (!graphics::SupportBackendAPI(api))
            continue;
          bool selected = api == request.graphics_api;
          if (ImGui::Selectable(graphics::BackendAPIString(api), selected) && !selected) {
            request.graphics_api = api;
            request.pipeline = sparkium::RENDER_PIPELINE_AUTO;
            submit();
          }
        }
        ImGui::EndCombo();
      }
      // A backend selection immediately invalidates the old capabilities.
      status = worker.Snapshot();
      const bool ready = !status.name.empty() && status.error.empty();
      ImGui::BeginDisabled(!ready);
      const auto pipeline = request.pipeline.value_or(status.pipeline);
      const std::string auto_label = std::string("Auto (") + PipelineName(status.automatic_pipeline) + ")";
      const char *pipeline_label =
          pipeline == sparkium::RENDER_PIPELINE_AUTO ? auto_label.c_str() : PipelineName(pipeline);
      if (ImGui::BeginCombo("Pipeline", pipeline_label)) {
        const bool compute_backend =
            status.backend == sparkium::RenderBackend::CPU || status.backend == sparkium::RenderBackend::CUDA;
        for (auto option : {sparkium::RENDER_PIPELINE_AUTO, sparkium::RENDER_PIPELINE_RASTERIZATION,
                            sparkium::RENDER_PIPELINE_RAY_TRACING, sparkium::RENDER_PIPELINE_RT_FALLBACK,
                            sparkium::RENDER_PIPELINE_RAY_QUERY}) {
          if (option == sparkium::RENDER_PIPELINE_RASTERIZATION && compute_backend)
            continue;
          if (option == sparkium::RENDER_PIPELINE_RAY_TRACING && !status.ray_tracing)
            continue;
          if (option == sparkium::RENDER_PIPELINE_RAY_QUERY && !status.ray_query)
            continue;
          const bool current = option == pipeline;
          const char *label = option == sparkium::RENDER_PIPELINE_AUTO ? auto_label.c_str() : PipelineName(option);
          if (ImGui::Selectable(label, current) && pipeline != option) {
            request.pipeline = option;
            submit();
          }
          if (current)
            ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
      }
      int samples = request.samples.value_or(status.samples);
      if (ImGui::SliderInt("Samples / dispatch", &samples, 1, 256)) {
        request.samples = samples;
        submit();
      }
      if (ImGui::Button("Reset film"))
        submit();
      ImGui::EndDisabled();
      ImGui::SameLine();
      if (ImGui::Button("Reload")) {
        ++request.reload;
        submit();
      }
      ImGui::Text("%s", scene_files[selected].string().c_str());
      ImGui::Text("Active render backend: %s",
                  sparkium::BackendName(sparkium::BackendSelection{status.backend, status.graphics_api}));
      ImGui::Text("Display backend: %s", graphics::BackendAPIString(display->API()));
      if (!status.device.empty())
        ImGui::Text("Render device: %s", status.device.c_str());
      ImGui::Text("Pipeline: %s", PipelineName(status.resolved_pipeline));
      if (status.updating)
        ImGui::TextUnformatted("Loading / rendering...");
      if (status.frame) {
        const auto &frame = *status.frame;
        ImGui::Text("Render: %.1f ms / dispatch", frame.render_seconds * 1000);
        if (status.resolved_pipeline != sparkium::RENDER_PIPELINE_RASTERIZATION) {
          const double rays =
              frame.render_seconds > 0 ? double(frame.width) * frame.height * status.samples / frame.render_seconds : 0;
          ImGui::Text("Camera rays/s: %.2f M", rays / 1e6);
          if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Width x height x samples per dispatch / render time (independent of GUI FPS).");
          ImGui::Text("Accumulated spp: %d", frame.accumulated_samples);
        }
      }
      if (!status.error.empty())
        ImGui::TextColored({1, .3f, .3f, 1}, "%s", status.error.c_str());
      ImGui::Text("GUI: %.1f FPS", fps_counter.TickFPS());
      ImGui::End();
      window->EndImGuiFrame();

      std::unique_ptr<graphics::CommandContext> command_context;
      display->CreateCommandContext(&command_context);
      command_context->CmdPresent(window.get(), image.get());
      display->SubmitCommandContext(command_context.get());
      if (frame_limit && status.finished && worker.Snapshot().revision == status.revision &&
          (!status.error.empty() || status.frame == shown)) {
        if (!status.error.empty()) {
          std::cerr << "sparkium_gui: " << status.error << '\n';
          exit_code = 1;
        } else if (shown) {
          std::cout << "Rendered " << shown->number << " frames with "
                    << sparkium::BackendName(sparkium::BackendSelection{status.backend, status.graphics_api})
                    << "; displayed via " << graphics::BackendAPIString(display->API()) << '\n';
        }
        break;
      }
      const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - ui_start).count();
      if (elapsed < 1.0 / 60.0)
        glfwWaitEventsTimeout(1.0 / 60.0 - elapsed);
    }
    display->WaitGPU();
    window->TerminateImGui();
    // Remove the window before waiting for an in-flight render dispatch to end.
    window.reset();
    worker.Stop();
    return exit_code;
  } catch (const std::exception &exception) {
    std::cerr << "sparkium_gui: " << exception.what() << '\n';
    return 1;
  }
}
