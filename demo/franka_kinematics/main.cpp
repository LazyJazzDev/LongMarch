#include <imgui.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>

#include "../sparkium_backend.h"
#include "franka_scene.h"
#include "sparkium/core/render_worker.h"
#include "sparkium/sparkium.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace grassland;

int main(int argc, char **argv) {
  try {
    auto backend = rhi::BACKEND_API_DEFAULT;
    bool headless = false;
    FrankaScene::Settings settings;
    int frames = 0;
    std::filesystem::path output{"output.bmp"};
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        std::cout
            << "Franka Kinematics\n--backend auto|metal|vulkan|d3d12 --frames N --headless --raster --output image.bmp\n";
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
      else if (arg == "--raster")
        settings.ray_tracing = false;
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
    constexpr int width = 1024, height = 512;
    int sample_limit = frames * (settings.ray_tracing ? 32 : 1);
    rhi::ObjPtr<rhi::IQueue> queue = device->GetQueue(rhi::QUEUE_TYPE_MAIN);
    if (!queue)
      return 1;
    rhi::ObjPtr<rhi::IWindow> window;
    if (!headless) {
      window = device->CreateGLFWWindow({width, height, "Franka Kinematics", rhi::PRESENT_MODE_CONTINUOUS});
      if (!window || !window->EnableImGui(true, nullptr, 20.0f))
        return 1;
    }
    using Worker = sparkium::RenderWorker<FrankaScene>;
    Worker worker([device = device.Get()] { return FrankaScene::Create(device); }, settings, sample_limit, false, 8);
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
              if (ImGui::Begin("Franka Joint Control", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                changed |= ImGui::Checkbox("Ray Tracing", &settings.ray_tracing);
                if (settings.ray_tracing && !device->SupportRayTracingPipeline() && !device->SupportRayQuery())
                  ImGui::Text("Ray Tracing not supported on this device!");
                ImGui::Separator();
                if (ImGui::Button("Center")) {
                  for (size_t i = 0; i < settings.joints.size(); ++i)
                    settings.joints[i] = (FrankaScene::joint_lower[i] + FrankaScene::joint_upper[i]) * 0.5f;
                  changed = true;
                }
                for (size_t i = 0; i < settings.joints.size(); ++i)
                  changed |= ImGui::SliderFloat(("Joint " + std::to_string(i + 1)).c_str(), &settings.joints[i],
                                                FrankaScene::joint_lower[i], FrankaScene::joint_upper[i]);
                ImGui::Text("Samples: %d", state.completed_samples);
              }
              ImGui::End();
              if (changed) {
                sample_limit = frames * (settings.ray_tracing ? 32 : 1);
                requested_revision = worker.Configure(settings, sample_limit);
              }
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
      if (!stbi_write_bmp(output.string().c_str(), width, height, 4, pixels.data())) {
        std::cerr << "cannot save output image\n";
        return 1;
      }
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "franka_kinematics: " << error.what() << '\n';
    return 1;
  }
}
