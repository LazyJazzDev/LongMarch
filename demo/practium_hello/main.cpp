#include <imgui.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>

#include "../sparkium_backend.h"
#include "practium_scene.h"
#include "sparkium/core/render_worker.h"
#include "sparkium/sparkium.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace grassland;

int main(int argc, char **argv) {
  try {
    auto backend = rhi::BACKEND_API_DEFAULT;
    bool headless = false;
    PractiumDemoScene::Settings settings;
    int frames = 0;
    std::filesystem::path output;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        std::cout
            << "Practium Hello\n--backend auto|metal|vulkan|d3d12 --frames N --headless --ray-tracing --run --output image.png\n";
        return 0;
      } else if (arg == "--backend" && i + 1 < argc)
        backend = ParseSparkiumBackend(argv[++i]);
      else if (arg == "--frames" && i + 1 < argc) {
        frames = std::stoi(argv[++i]);
        if (frames <= 0) {
          std::cerr << "--frames must be positive\n";
          return 2;
        }
      } else if (arg == "--headless")
        headless = true;
      else if (arg == "--ray-tracing")
        settings.ray_tracing = true;
      else if (arg == "--run")
        settings.pause = false;
      else if (arg == "--output" && i + 1 < argc)
        output = argv[++i];
      else {
        std::cerr << "unknown or incomplete argument: " << arg << '\n';
        return 2;
      }
    }
    if (frames > std::numeric_limits<int>::max() / 32) {
      std::cerr << "--frames is too large\n";
      return 2;
    }
    if (headless && frames == 0) {
      std::cerr << "--headless requires --frames\n";
      return 2;
    }

    rhi::ObjPtr<rhi::ICore> rhi_core = rhi::CreateCore(backend);
    if (!rhi_core)
      return 1;
    rhi::ObjPtr<rhi::IDevice> device = rhi_core->CreateDeviceByAutoSelection();
    if (!device)
      return 1;
    constexpr int width = 640, height = 480;
    int sample_limit = frames * (settings.ray_tracing ? 32 : 1);
    rhi::ObjPtr<rhi::IQueue> queue = device->GetQueue(rhi::QUEUE_TYPE_MAIN);
    if (!queue)
      return 1;
    int ui_redraw_frames = 2;
    rhi::ObjPtr<rhi::IWindow> window;
    if (!headless) {
      window = device->CreateGLFWWindow({width, height, "Practium Hello", rhi::PRESENT_MODE_CONTINUOUS});
      if (!window || !window->EnableImGui(true, nullptr, 20.0f))
        return 1;
      auto invalidate_ui = [&ui_redraw_frames](auto...) { ui_redraw_frames = 2; };
      window->ResizeEvent().RegisterCallback(invalidate_ui);
      window->FramebufferResizeEvent().RegisterCallback(invalidate_ui);
      window->FocusEvent().RegisterCallback(invalidate_ui);
      window->CursorEnterEvent().RegisterCallback(invalidate_ui);
      window->MouseMoveEvent().RegisterCallback(invalidate_ui);
      window->MouseButtonEvent().RegisterCallback(invalidate_ui);
      window->ScrollEvent().RegisterCallback(invalidate_ui);
      window->KeyEvent().RegisterCallback(invalidate_ui);
      window->CharEvent().RegisterCallback(invalidate_ui);
    }
    using Worker = sparkium::RenderWorker<PractiumDemoScene>;
    Worker worker([device = device.Get()] { return PractiumDemoScene::Create(device); }, settings, sample_limit, false,
                  settings.ray_tracing ? 32 : 1,
                  [interactive = !headless] {
                    if (interactive)
                      glfwPostEmptyEvent();
                  });
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
              bool changed = false;
              ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Once);
              ImGui::SetNextWindowBgAlpha(0.3f);
              ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
              changed |= ImGui::Checkbox("Day Light", &settings.day_light);
              changed |= ImGui::Checkbox("Pause", &settings.pause);
              changed |= ImGui::Checkbox("Ray Tracing", &settings.ray_tracing);
              if (settings.ray_tracing && !device->SupportRayTracingPipeline() && !device->SupportRayQuery())
                ImGui::Text("Ray Tracing not supported on this device!");
              if (ImGui::Button("Reset")) {
                ++settings.reset_sequence;
                changed = true;
              }
              ImGui::End();
              if (changed) {
                sample_limit = frames * (settings.ray_tracing ? 32 : 1);
                requested_revision = worker.Configure(settings, sample_limit, false, settings.ray_tracing ? 32 : 1);
              }
              if (changed || ImGui::IsAnyItemActive())
                ui_redraw_frames = 2;
            }))
          return 1;
        if (window->IsSwapChainInvalid()) {
          ui_redraw_frames = 2;
          if (!window->RebuildSwapChain())
            return 1;
        }
        if (window->IsSwapChainInvalid()) {
          glfwWaitEventsTimeout(0.01);
          continue;
        }
        auto latest = worker.Latest();
        // Input keeps the UI responsive while an unchanged completed image
        // does not need another cross-queue presentation submission.
        if (!ui_redraw_frames && latest == frame) {
          // An idle poll must not postpone a newly completed image or input.
          // Only an actual presentation advances the pacing deadline.
          glfwWaitEventsTimeout(std::chrono::duration<double>(presentation_interval).count());
          continue;
        }
        frame = std::move(latest);
        rhi::ObjPtr<rhi::ICommandContext> commands = queue->CreateCommandContext();
        if (!commands)
          return 1;
        rhi::SubmitInfo dependencies;
        if (frame) {
          dependencies.waits.push_back(frame->rendered->GetEvent());
          if (!commands->CmdPresentImage(window.Get(), frame->Image(), rhi::COLOR_SPACE_SRGB_NONLINEAR))
            return 1;
        } else {
          rhi::ObjPtr<rhi::IImage> background = window->GetNextImage();
          if (!background || !commands->CmdClearImage(background.Get(), {0.0f, 0.0f, 0.0f, 1.0f}) ||
              !commands->CmdPresent(window.Get()))
            return 1;
        }
        completion = commands->Submit(dependencies);
        if (!completion)
          return 1;
        if (frame)
          worker.Presented(frame, completion);
        if (ui_redraw_frames)
          --ui_redraw_frames;
        ++ui_index;
        next_presentation = std::chrono::steady_clock::now() + presentation_interval;
      }
    }
    if (window && window->ShouldClose())
      glfwHideWindow(window->GLFWWindow());
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
      if (!stbi_write_png(output.string().c_str(), width, height, 4, pixels.data(), width * 4)) {
        std::cerr << "cannot save output image\n";
        return 1;
      }
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "practium_hello: " << error.what() << '\n';
    return 1;
  }
}
