#include <imgui.h>

#include <chrono>
#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <limits>

#include "../sparkium_backend.h"
#include "sparkium/core/render_worker.h"
#include "sparkium/sparkium.h"
#include "thin_lens_scene.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace grassland;

int main(int argc, char **argv) {
  try {
    auto backend = rhi::BACKEND_API_DEFAULT;
    bool headless = false;
    ThinLensScene::Settings settings;
    int frames = 0;
    std::filesystem::path output;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        std::cout << "Thin Lens: interactive depth of field and bokeh\n"
                     "--backend auto|metal|vulkan|d3d12 --frames N --headless --output image.png\n"
                     "--pinhole --focus distance --aperture radius --blades N (0 = circle)\n"
                     "--geometry-offset cutoff (0..1, default 0.1; 0 disables)\n";
        return 0;
      } else if (arg == "--backend" && i + 1 < argc)
        backend = ParseSparkiumBackend(argv[++i]);
      else if (arg == "--frames" && i + 1 < argc) {
        frames = std::stoi(argv[++i]);
        if (frames <= 0)
          throw std::invalid_argument("--frames must be positive");
      } else if (arg == "--headless")
        headless = true;
      else if (arg == "--pinhole")
        settings.pinhole_mode = true;
      else if (arg == "--output" && i + 1 < argc)
        output = argv[++i];
      else if (arg == "--focus" && i + 1 < argc)
        settings.focus_distance = std::stof(argv[++i]);
      else if (arg == "--aperture" && i + 1 < argc)
        settings.aperture_radius = std::stof(argv[++i]);
      else if (arg == "--geometry-offset" && i + 1 < argc)
        settings.geometry_offset = std::stof(argv[++i]);
      else if (arg == "--blades" && i + 1 < argc)
        settings.aperture_blades = std::stoi(argv[++i]);
      else
        throw std::invalid_argument("unknown or incomplete argument: " + arg);
    }
    if (frames > std::numeric_limits<int>::max() / 8)
      throw std::invalid_argument("--frames is too large");
    if (headless && frames == 0)
      throw std::invalid_argument("--headless requires --frames");

    rhi::ObjPtr<rhi::ICore> rhi_core = rhi::CreateCore(backend);
    if (!rhi_core)
      return 1;
    rhi::ObjPtr<rhi::IDevice> device = rhi_core->CreateDeviceByAutoSelection();
    if (!device)
      return 1;
    constexpr int width = 1100, height = 700;
    const int sample_limit = frames * 8;
    rhi::ObjPtr<rhi::IQueue> queue = device->GetQueue(rhi::QUEUE_TYPE_MAIN);
    if (!queue)
      return 1;
    rhi::ObjPtr<rhi::IWindow> window;
    if (!headless) {
      window = device->CreateGLFWWindow({width, height, "Sparkium - Thin Lens", rhi::PRESENT_MODE_CONTINUOUS});
      if (!window || !window->EnableImGui(true, nullptr, 18.0f))
        return 1;
    }
    const auto pipeline = device->SupportRayQuery()             ? sparkium::RENDER_PIPELINE_RAY_QUERY
                          : device->SupportRayTracingPipeline() ? sparkium::RENDER_PIPELINE_RAY_TRACING
                                                                : sparkium::RENDER_PIPELINE_RT_FALLBACK;
    std::cout << "Device: " << device->Name() << ", pipeline: " << int(pipeline) << std::endl;
    using Worker = sparkium::RenderWorker<ThinLensScene>;
    Worker worker([device = device.Get()] { return ThinLensScene::Create(device); }, settings, sample_limit, false, 8);
    uint64_t requested_revision = 1;
    std::shared_ptr<const Worker::Frame> frame;
    std::array<rhi::ObjPtr<rhi::ISubmissionToken>, 2> ui_pending;
    size_t ui_index{};
    const auto *mode = window ? glfwGetVideoMode(glfwGetPrimaryMonitor()) : nullptr;
    const int refresh_rate = mode && mode->refreshRate > 0 ? mode->refreshRate : 60;
    const auto presentation_interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / refresh_rate));
    auto next_presentation = std::chrono::steady_clock::now();
    if (headless) {
      frame = worker.WaitForFrame(requested_revision, sample_limit);
      if (!frame) {
        std::cerr << worker.GetState().error << '\n';
        return 1;
      }
    } else {
      while (!window->ShouldClose() &&
             (!sample_limit || !frame || frame->revision != requested_revision || frame->samples < sample_limit)) {
        if (!rhi::PollEvents())
          return 1;
        const auto state = worker.GetState();
        if (!state.error.empty()) {
          std::cerr << state.error << '\n';
          return 1;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now < next_presentation) {
          glfwWaitEventsTimeout(std::min(0.01, std::chrono::duration<double>(next_presentation - now).count()));
          continue;
        }
        auto &completion = ui_pending[ui_index % ui_pending.size()];
        if (completion && completion->GetStatus() == rhi::SUBMISSION_STATUS_PENDING) {
          glfwWaitEventsTimeout(0.001);
          continue;
        }
        if (completion && completion->GetStatus() != rhi::SUBMISSION_STATUS_COMPLETED)
          return 1;
        if (!window->UpdateImGui([&] {
              ImGui::GetIO().IniFilename = nullptr;
              ImGui::SetNextWindowPos({16, 16}, ImGuiCond_Always);
              ImGui::SetNextWindowSize({330, 0}, ImGuiCond_Always);
              ImGui::SetNextWindowBgAlpha(0.92f);
              ImGui::Begin("Thin Lens", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove);
              ImGui::TextWrapped("Focus one subject. Watch the background lights change into aperture-shaped bokeh.");
              ImGui::PushItemWidth(160.0f);
              bool changed = ImGui::Checkbox("Pinhole comparison", &settings.pinhole_mode);
              ImGui::BeginDisabled(settings.pinhole_mode);
              changed |= ImGui::SliderFloat("Focus", &settings.focus_distance, 2.5f, 13.0f, "%.2f");
              if (ImGui::Button("Near / orange")) {
                settings.focus_distance = 3.4f;
                changed = true;
              }
              ImGui::SameLine();
              if (ImGui::Button("Mid / teal")) {
                settings.focus_distance = 5.4f;
                changed = true;
              }
              if (ImGui::Button("Far / blue")) {
                settings.focus_distance = 8.4f;
                changed = true;
              }
              changed |= ImGui::SliderFloat("Aperture radius", &settings.aperture_radius, 0.0f, 0.45f, "%.3f");
              int shape = settings.aperture_blades == 0 ? 0 : settings.aperture_blades - 2;
              if (ImGui::Combo("Shape", &shape, "Circle\0Triangle\0Square\0Pentagon\0Hexagon\0Heptagon\0Octagon\0")) {
                settings.aperture_blades = shape == 0 ? 0 : shape + 2;
                changed = true;
              }
              changed |= ImGui::SliderAngle("Rotation", &settings.aperture_rotation, 0, 180);
              changed |= ImGui::SliderFloat("Horizontal ratio", &settings.aperture_ratio, 0.4f, 2.5f, "%.2f");
              ImGui::EndDisabled();
              if (ImGui::SliderFloat("Geometry offset", &settings.geometry_offset, 0.0f, 1.0f, "%.3f")) {
                changed = true;
              }
              if (ImGui::Button("Reset lens")) {
                settings.pinhole_mode = false;
                settings.aperture_radius = 0.22f;
                settings.focus_distance = 5.4f;
                settings.aperture_blades = 6;
                settings.aperture_rotation = 0;
                settings.aperture_ratio = 1;
                changed = true;
              }
              if (changed)
                requested_revision = worker.Configure(settings, sample_limit);
              ImGui::Text("Samples: %d", state.completed_samples);
              ImGui::TextWrapped("Larger aperture = shallower depth of field. Zero aperture matches a pinhole.");
              ImGui::PopItemWidth();
              ImGui::End();
            }))
          return 1;
        if (window->IsSwapChainInvalid() && !window->RebuildSwapChain())
          return 1;
        if (window->IsSwapChainInvalid())
          continue;
        frame = worker.Latest();
        rhi::ObjPtr<rhi::ICommandContext> commands = queue->CreateCommandContext();
        if (!commands)
          return 1;
        rhi::SubmitInfo dependencies;
        if (frame) {
          dependencies.waits.push_back(frame->rendered->GetEvent());
          if (!commands->CmdPresentImage(window.Get(), frame->Image(), rhi::COLOR_SPACE_SRGB_NONLINEAR))
            return 1;
        } else if (!commands->CmdPresent(window.Get()))
          return 1;
        completion = commands->Submit(dependencies);
        if (!completion)
          return 1;
        if (frame)
          worker.Presented(frame, completion);
        ++ui_index;
        next_presentation = std::chrono::steady_clock::now() + presentation_interval;
      }
    }
    worker.Stop();
    if (const auto state = worker.GetState(); !state.error.empty()) {
      std::cerr << state.error << '\n';
      return 1;
    }
    frame = worker.Latest();
    for (const auto &completion : ui_pending)
      if (completion && !completion->WaitHost())
        return 1;
    if (!output.empty()) {
      if (!frame)
        return 1;
      if (output.has_parent_path())
        std::filesystem::create_directories(output.parent_path());
      std::vector<uint8_t> pixels(width * height * 4);
      rhi::ObjPtr<rhi::ICommandContext> commands = queue->CreateCommandContext();
      if (!commands)
        return 1;
      rhi::ObjPtr<rhi::IDownloadToken> readback = commands->CmdDownloadImage(frame->Image());
      rhi::SubmitInfo dependencies;
      dependencies.waits.push_back(frame->rendered->GetEvent());
      rhi::ObjPtr<rhi::ISubmissionToken> downloaded = commands->Submit(dependencies);
      if (!readback || !downloaded || !downloaded->WaitHost() || !readback->CopyTo(pixels.data(), pixels.size()))
        return 1;
      if (!stbi_write_png(output.string().c_str(), width, height, 4, pixels.data(), width * 4))
        throw std::runtime_error("cannot save output image");
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "thin_lens: " << error.what() << '\n';
    return 1;
  }
}
