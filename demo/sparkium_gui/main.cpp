#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <iostream>

#include "../sparkium_backend.h"
#include "browser_renderer.h"

using namespace grassland;

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

sparkium::RenderPipeline Resolve(rhi::IDevice *device, sparkium::RenderPipeline pipeline) {
  if (pipeline == sparkium::RENDER_PIPELINE_AUTO)
    return device->SupportRayQuery()             ? sparkium::RENDER_PIPELINE_RAY_QUERY
           : device->SupportRayTracingPipeline() ? sparkium::RENDER_PIPELINE_RAY_TRACING
                                                 : sparkium::RENDER_PIPELINE_RT_FALLBACK;
  if (pipeline == sparkium::RENDER_PIPELINE_RAY_TRACING && !device->SupportRayTracingPipeline())
    return device->SupportRayQuery() ? sparkium::RENDER_PIPELINE_RAY_QUERY : sparkium::RENDER_PIPELINE_RT_FALLBACK;
  return pipeline;
}

void ResizeWindowForFilm(rhi::IWindow *window, uint32_t width, uint32_t height) {
  auto *native = window->GLFWWindow();
  auto *monitor = glfwGetPrimaryMonitor();
  int x{}, y{}, work_width{}, work_height{}, left{}, top{}, right{}, bottom{};
  glfwGetMonitorWorkarea(monitor, &x, &y, &work_width, &work_height);
  glfwGetWindowFrameSize(native, &left, &top, &right, &bottom);
  const int available_width = std::max(1, work_width - left - right);
  const int available_height = std::max(1, work_height - top - bottom);
  const float scale = std::min({1.0f, float(available_width) / width, float(available_height) / height});
  const int target_width = std::max(1, int(std::floor(width * scale)));
  const int target_height = std::max(1, int(std::floor(height * scale)));
  glfwSetWindowSize(native, target_width, target_height);
  glfwSetWindowPos(native, x + left + (available_width - target_width) / 2,
                   y + top + (available_height - target_height) / 2);
}
}  // namespace

int main(int argc, char **argv) {
  try {
    std::filesystem::path input = FindAssetPath("scenes");
    bool input_selected = false;
    bool hdr_requested = false;
    auto backend = rhi::BACKEND_API_DEFAULT;
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

    rhi::ObjPtr<rhi::ICore> rhi_core = rhi::CreateCore(backend);
    if (!rhi_core)
      return 1;
    rhi::ObjPtr<rhi::IDevice> device = rhi_core->CreateDeviceByAutoSelection();
    if (!device)
      return 1;
    rhi::WindowDesc description;
    description.width = 1280;
    description.height = 720;
    description.title = "Sparkium Scene Browser";
    description.present_mode = rhi::PRESENT_MODE_CONTINUOUS;
    rhi::ObjPtr<rhi::IWindow> window = device->CreateGLFWWindow(description);
    if (!window || !window->EnableImGui(true, nullptr, 18.0f))
      return 1;
    rhi::ObjPtr<rhi::IQueue> main_queue = device->GetQueue(rhi::QUEUE_TYPE_MAIN);
    if (!main_queue)
      return 1;
    BrowserRenderer renderer(device.Get());
    BrowserRenderer::Settings settings;
    bool hdr_active = false;
    std::string hdr_error;
    size_t selected = 0;
    bool loading = true;
    uint64_t observed_revision{};
    uint32_t displayed_width{}, displayed_height{};
    uint64_t requested_revision = renderer.Load(scene_files[selected], hdr_active);
    FPSCounter fps_counter;
    bool show_browser = true;
    int presented_frames{};
    std::array<rhi::ObjPtr<rhi::ISubmissionToken>, 2> ui_pending;
    size_t ui_index{};
    while (!window->ShouldClose() && (!frame_limit || presented_frames < frame_limit)) {
      if (!rhi::PollEvents())
        return 1;
      if (hdr_requested != hdr_active) {
        bool changed = false;
        if (!hdr_requested)
          changed = window->SetColorSpace(rhi::COLOR_SPACE_SRGB_NONLINEAR);
        else {
          for (const auto &option : window->GetSupportedPresentationOptions())
            if (option.color_space == rhi::COLOR_SPACE_EXTENDED_SRGB_LINEAR ||
                option.color_space == rhi::COLOR_SPACE_HDR10_ST2084)
              if (window->SetColorSpace(option.color_space)) {
                changed = true;
                break;
              }
        }
        if (changed) {
          hdr_active = hdr_requested;
          settings.hdr = hdr_active;
          hdr_error.clear();
          if (loading)
            requested_revision = renderer.Load(scene_files[selected], hdr_active);
          else
            requested_revision = renderer.Configure(settings, false);
        } else {
          hdr_error = "Requested presentation mode is unavailable.";
          hdr_requested = hdr_active;
        }
      }
      auto state = renderer.GetState();
      if (state.revision >= requested_revision && state.revision != observed_revision) {
        observed_revision = state.revision;
        settings = state.settings;
        settings.hdr = hdr_active;
        loading = false;
        if (state.width && state.height && (state.width != displayed_width || state.height != displayed_height)) {
          ResizeWindowForFilm(window.Get(), state.width, state.height);
          displayed_width = state.width;
          displayed_height = state.height;
        }
      }
      bool changed = false, reset = false, reload = false;
      if (!window->UpdateImGui([&] {
            ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
            ImGui::SetNextWindowBgAlpha(hdr_active ? 1.0f : 0.85f);
            ImGui::Begin("Sparkium scenes", &show_browser, ImGuiWindowFlags_AlwaysAutoResize);
            if (ImGui::BeginCombo("Scene", loading ? "Loading..." : state.name.c_str())) {
              for (size_t i = 0; i < scene_files.size(); ++i) {
                const bool current = i == selected;
                if (ImGui::Selectable(scene_files[i].parent_path().filename().string().c_str(), current)) {
                  selected = i;
                  reload = true;
                }
                if (current)
                  ImGui::SetItemDefaultFocus();
              }
              ImGui::EndCombo();
            }
            ImGui::BeginDisabled(loading || !state.error.empty());
            const auto resolved = Resolve(device.Get(), settings.pipeline);
            const std::string automatic =
                std::string("Auto (") + PipelineName(Resolve(device.Get(), sparkium::RENDER_PIPELINE_AUTO)) + ")";
            if (ImGui::BeginCombo("Pipeline", settings.pipeline == sparkium::RENDER_PIPELINE_AUTO
                                                  ? automatic.c_str()
                                                  : PipelineName(resolved))) {
              for (auto option : {sparkium::RENDER_PIPELINE_AUTO, sparkium::RENDER_PIPELINE_RASTERIZATION,
                                  sparkium::RENDER_PIPELINE_RAY_TRACING, sparkium::RENDER_PIPELINE_RT_FALLBACK,
                                  sparkium::RENDER_PIPELINE_RAY_QUERY}) {
                if (option == sparkium::RENDER_PIPELINE_RAY_TRACING && !device->SupportRayTracingPipeline())
                  continue;
                if (option == sparkium::RENDER_PIPELINE_RAY_QUERY && !device->SupportRayQuery())
                  continue;
                if (ImGui::Selectable(
                        option == sparkium::RENDER_PIPELINE_AUTO ? automatic.c_str() : PipelineName(option),
                        option == settings.pipeline)) {
                  reset |= settings.pipeline != option;
                  settings.pipeline = option;
                }
              }
              ImGui::EndCombo();
            }
            ImGui::Checkbox("HDR preview", &hdr_requested);
            if (!hdr_error.empty())
              ImGui::TextWrapped("HDR unavailable: %s", hdr_error.c_str());
            changed |= ImGui::SliderFloat("Exposure (EV)", &settings.film.exposure, -8.0f, 8.0f, "%.2f");
            ImGui::TextUnformatted(hdr_active ? "Display: HDR" : "Display: SDR (scene view transform)");
            if (hdr_active) {
              float nits{};
              if (window->GetHDRReferenceWhiteNits(nits))
                ImGui::Text("Reference white: %.0f nits (%.2fx)", nits, window->GetHDRReferenceWhiteScale());
              else
                ImGui::TextUnformatted("Reference white: unavailable");
              ImGui::Text("HDR headroom: %.2fx", window->GetHDRHeadroom());
            }
            if (ImGui::CollapsingHeader("Render settings", ImGuiTreeNodeFlags_DefaultOpen)) {
              if (resolved == sparkium::RENDER_PIPELINE_RASTERIZATION)
                reset |=
                    ImGui::ColorEdit3("Ambient light", &settings.raster.ambient_light.x, ImGuiColorEditFlags_Float);
              else {
                reset |= ImGui::SliderInt("Samples / batch", &settings.ray.samples_per_dispatch, 1, 4, "%d",
                                          ImGuiSliderFlags_AlwaysClamp);
                reset |= ImGui::SliderInt("Max bounces", &settings.ray.max_bounces, 1, 128, "%d",
                                          ImGuiSliderFlags_AlwaysClamp);
                bool alpha_shadow = settings.ray.alpha_shadow != 0;
                if (ImGui::Checkbox("Alpha shadows", &alpha_shadow)) {
                  settings.ray.alpha_shadow = alpha_shadow;
                  reset = true;
                }
                reset |= ImGui::ColorEdit3("Background", &settings.ray.background_color.x, ImGuiColorEditFlags_Float);
                reset |= ImGui::SliderFloat("Persistence", &settings.film.persistence, 0.0f, 1.0f, "%.3f",
                                            ImGuiSliderFlags_AlwaysClamp);
                reset |= ImGui::SliderFloat("Sample clamp", &settings.film.clamping, 0.01f, 10000.0f, "%.2f",
                                            ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
                reset |= ImGui::SliderFloat("Max exposure", &settings.film.max_exposure, 0.01f, 10000.0f, "%.2f",
                                            ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
                if (hdr_requested && settings.film.max_exposure <= 1.0f)
                  ImGui::TextWrapped("Max exposure <= 1 clips scene highlights before HDR display.");
              }
            }
            if (ImGui::CollapsingHeader("View settings")) {
              changed |=
                  ImGui::Combo("View transform", &settings.film.view_transform, "Normalized\0Standard\0Filmic\0");
              ImGui::BeginDisabled(!hdr_active && settings.film.view_transform != 2);
              changed |=
                  ImGui::SliderFloat("Gamma", &settings.film.gamma, 0.1f, 4.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
              changed |= ImGui::SliderFloat("Contrast", &settings.film.contrast, 0.0f, 4.0f, "%.2f",
                                            ImGuiSliderFlags_AlwaysClamp);
              ImGui::EndDisabled();
            }
            if (ImGui::Button("Reset film"))
              reset = true;
            ImGui::EndDisabled();
            ImGui::SameLine();
            if (ImGui::Button("Reload"))
              reload = true;
            ImGui::Text("%s", scene_files[selected].string().c_str());
            ImGui::Text("Backend: %s", rhi::BackendAPIName(backend).c_str());
            ImGui::Text("Resolution: %u x %u", state.width, state.height);
            ImGui::Text("Pipeline: %s", PipelineName(resolved));
            if (resolved != sparkium::RENDER_PIPELINE_RASTERIZATION)
              ImGui::Text("Completed spp: %d", state.completed_samples);
            if (!state.error.empty())
              ImGui::TextColored({1, .3f, .3f, 1}, "%s", state.error.c_str());
            ImGui::Text("%.1f FPS", fps_counter.TickFPS());
            ImGui::End();
          }))
        return 1;
      if (reload) {
        requested_revision = renderer.Load(scene_files[selected], hdr_active);
        loading = true;
      } else if (changed || reset)
        requested_revision = renderer.Configure(settings, reset);
      if (window->IsSwapChainInvalid() && !window->RebuildSwapChain())
        return 1;
      if (window->IsSwapChainInvalid())
        continue;
      auto &pending = ui_pending[ui_index++ % ui_pending.size()];
      if (pending && !pending->WaitHost())
        return 1;
      auto frame = renderer.Latest();
      rhi::ObjPtr<rhi::ICommandContext> commands = main_queue->CreateCommandContext();
      if (!commands)
        return 1;
      rhi::SubmitInfo dependencies;
      if (frame) {
        dependencies.waits.push_back(frame->rendered->GetEvent());
        if (!commands->CmdPresentImage(window.Get(), frame->image.Get(),
                                       frame->hdr ? rhi::COLOR_SPACE_BT709_LINEAR : rhi::COLOR_SPACE_SRGB_NONLINEAR))
          return 1;
      } else if (!commands->CmdPresent(window.Get()))
        return 1;
      pending = commands->Submit(dependencies);
      if (!pending)
        return 1;
      if (frame) {
        renderer.Presented(frame, pending);
        ++presented_frames;
      } else if (!state.error.empty() && frame_limit)
        return 1;
    }
    for (const auto &pending : ui_pending)
      if (pending && !pending->WaitHost())
        return 1;
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "sparkium_gui: " << exception.what() << '\n';
    return 1;
  }
}
