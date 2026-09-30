#include <long_march.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

#include "../sparkium_backend.h"
#include "snowberg/gui/gui.h"

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
  const auto work = window->GetMonitorWorkArea();
  const auto frame = window->GetFrameSize();
  const int work_x = work.x, work_y = work.y, work_width = work.z, work_height = work.w;
  const int frame_left = frame.x, frame_top = frame.y, frame_right = frame.z, frame_bottom = frame.w;
  const int available_width = std::max(1, work_width - frame_left - frame_right);
  const int available_height = std::max(1, work_height - frame_top - frame_bottom);
  const float scale = std::min({1.0f, static_cast<float>(available_width) / film->GetWidth(),
                                static_cast<float>(available_height) / film->GetHeight()});
  const int target_width = std::max(1, static_cast<int>(std::floor(film->GetWidth() * scale)));
  const int target_height = std::max(1, static_cast<int>(std::floor(film->GetHeight() * scale)));

  window->Resize(target_width, target_height);
  window->SetPosition(work_x + frame_left + (available_width - target_width) / 2,
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
    bool hdr_active = false;
    std::string hdr_error;
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
    if (hdr_requested) {
      if (window->SetHDR(true) == 0) {
        hdr_active = true;
        create_display_image();
      } else {
        hdr_error = "Requested HDR presentation mode is unavailable; see the application log.";
        hdr_requested = false;
        std::cerr << "HDR unavailable; continuing in SDR: " << hdr_error << '\n';
      }
    }

    snowberg::gui::Context ui(graphics_core.get(), window.get(), snowberg::gui::DefaultFont());
    std::cout << "Display: " << (hdr_active ? "HDR" : "SDR") << std::endl;
    FPSCounter fps_counter;
    int ui_page = 0;

    int rendered_frames = 0;
    while (!window->ShouldClose() && (!frame_limit || rendered_frames++ < frame_limit)) {
      // Apply before building the UI so the scene and overlay use the same format.
      if (hdr_requested != hdr_active) {
        if (window->SetHDR(hdr_requested) == 0) {
          hdr_active = hdr_requested;
          hdr_error.clear();
          create_display_image();
        } else {
          hdr_error = "Could not change HDR presentation; see the application log.";
          hdr_requested = hdr_active;
        }
      }

      ui.BeginFrame();
      ui.BeginPanel("sparkium", 12, 12, 370, "Sparkium");
      ui.Choice("Page", &ui_page, {"Scene", "Sampling", "Environment", "Display", "Stats"});
      ui.Separator();
      auto *film = loaded->GetFilm();
      auto &settings = loaded->GetScene()->settings;
      if (ui_page == 0) {
        std::vector<std::string> scene_names;
        for (const auto &file : scene_files)
          scene_names.push_back(file.parent_path().filename().string());
        int scene_index = static_cast<int>(selected);
        if (ui.Choice("Scene", &scene_index, scene_names)) {
          const auto previous = selected;
          selected = static_cast<size_t>(scene_index);
          if (!load_selected())
            selected = previous;
          film = loaded->GetFilm();
        }
        const std::vector<sparkium::RenderPipeline> pipeline_options = {
            sparkium::RENDER_PIPELINE_AUTO, sparkium::RENDER_PIPELINE_RASTERIZATION,
            sparkium::RENDER_PIPELINE_RAY_TRACING, sparkium::RENDER_PIPELINE_RT_FALLBACK,
            sparkium::RENDER_PIPELINE_RAY_QUERY};
        std::vector<std::string> pipeline_names;
        std::vector<sparkium::RenderPipeline> available_pipelines;
        for (auto option : pipeline_options) {
          if (option == sparkium::RENDER_PIPELINE_RAY_TRACING && !graphics_core->DeviceRayTracingSupport())
            continue;
          if (option == sparkium::RENDER_PIPELINE_RAY_QUERY && !graphics_core->DeviceRayQuerySupport())
            continue;
          available_pipelines.push_back(option);
          pipeline_names.emplace_back(PipelineName(option));
        }
        int pipeline_index = 0;
        for (size_t i = 0; i < available_pipelines.size(); ++i)
          if (available_pipelines[i] == pipeline)
            pipeline_index = static_cast<int>(i);
        if (ui.Choice("Pipeline", &pipeline_index, pipeline_names)) {
          pipeline = available_pipelines[static_cast<size_t>(pipeline_index)];
          film->Reset();
        }
        if (ui.Button("Reload"))
          load_selected();
        if (ui.Button("Reset film"))
          film->Reset();
        ui.Text("Backend: " + std::string(graphics::BackendAPIString(graphics_core->API())));
        ui.Text("Resolution: " + std::to_string(film->GetWidth()) + " x " + std::to_string(film->GetHeight()));
      } else if (ui_page == 1) {
        const bool raster = core.ResolveRenderPipeline(pipeline) == sparkium::RENDER_PIPELINE_RASTERIZATION;
        bool reset = false;
        if (raster) {
          ui.Text("Rasterization uses one sample per frame.");
        } else {
          reset |= ui.Slider("Samples / frame", &settings.samples_per_dispatch, 1, 256);
          reset |= ui.Slider("Max bounces", &settings.max_bounces, 1, 128);
          bool alpha_shadow = settings.alpha_shadow != 0;
          if (ui.Checkbox("Alpha shadows", &alpha_shadow)) {
            settings.alpha_shadow = alpha_shadow;
            reset = true;
          }
          reset |= ui.Slider("Persistence", &film->info.persistence, 0.0f, 1.0f);
        }
        if (reset)
          film->Reset();
      } else if (ui_page == 2) {
        const bool raster = core.ResolveRenderPipeline(pipeline) == sparkium::RENDER_PIPELINE_RASTERIZATION;
        bool reset = false;
        if (raster) {
          reset |= ui.Slider("Ambient red", &settings.ambient_light.x, 0.0f, 2.0f);
          reset |= ui.Slider("Ambient green", &settings.ambient_light.y, 0.0f, 2.0f);
          reset |= ui.Slider("Ambient blue", &settings.ambient_light.z, 0.0f, 2.0f);
        } else {
          reset |= ui.Slider("Background red", &settings.background_color.x, 0.0f, 2.0f);
          reset |= ui.Slider("Background green", &settings.background_color.y, 0.0f, 2.0f);
          reset |= ui.Slider("Background blue", &settings.background_color.z, 0.0f, 2.0f);
          reset |= ui.Slider("Sample clamp", &film->info.clamping, 0.01f, 10000.0f);
          reset |= ui.Slider("Max exposure", &film->info.max_exposure, 0.01f, 10000.0f);
        }
        if (reset)
          film->Reset();
      } else if (ui_page == 3) {
        ui.Checkbox("HDR preview", &hdr_requested);
        if (!hdr_error.empty())
          ui.Text(hdr_error);
        ui.Slider("Exposure (EV)", &film->info.exposure, -8.0f, 8.0f);
        ui.Choice("View transform", &film->info.view_transform, {"Normalized", "Standard", "Filmic"});
        ui.Slider("Gamma", &film->info.gamma, 0.1f, 4.0f);
        ui.Slider("Contrast", &film->info.contrast, 0.0f, 4.0f);
        ui.Text(hdr_active ? "Display: HDR" : "Display: SDR");
        if (hdr_active) {
          const auto brightness = window->GetDisplayBrightness();
          if (brightness.sdr_white_nits > 0.0f)
            ui.Text("Reference white: " + std::to_string(int(brightness.sdr_white_nits)) + " nits");
          if (brightness.hdr_headroom > 0.0f)
            ui.Text("HDR headroom: " + std::to_string(brightness.hdr_headroom));
        }
      } else {
        const float fps = fps_counter.TickFPS();
        const auto resolved = core.ResolveRenderPipeline(pipeline);
        ui.Text("Pipeline: " + std::string(PipelineName(resolved)));
        ui.Text("FPS: " + std::to_string(fps));
        if (resolved != sparkium::RENDER_PIPELINE_RASTERIZATION) {
          const double rays = double(film->GetWidth()) * film->GetHeight() * settings.samples_per_dispatch * fps / 1e6;
          ui.Text("Camera ray/s: " + std::to_string(rays) + " M");
          ui.Text("Accumulated spp: " + std::to_string(film->info.accumulated_samples));
        }
        ui.Text(scene_files[selected].string());
        if (!load_error.empty())
          ui.Text(load_error);
      }
      ui.EndPanel();

      core.Render(loaded->GetScene(), loaded->GetCamera(), loaded->GetFilm(), pipeline);
      loaded->GetFilm()->Develop(image.get(), hdr_active);
      std::unique_ptr<graphics::CommandContext> command_context;
      graphics_core->CreateCommandContext(&command_context);
      auto *gui_image = ui.EndFrame(command_context.get(), image.get());

      command_context->CmdPresent(window.get(), gui_image);
      graphics_core->SubmitCommandContext(command_context.get());
      graphics::Window::PollEvents();
      if (resize_pending) {
        ResizeWindowForFilm(window.get(), loaded->GetFilm());
        resize_pending = false;
      }
    }
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "sparkium_gui: " << exception.what() << '\n';
    return 1;
  }
}
