#include "../../module.h"

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <thread>

#include "../../../rhi_common/last_frame.h"
#include "grassland/rhi/rhix/rhix.h"
#include "grassland/util/fps_counter.h"
#include "grassland/util/virtual_file_system.h"
#include "imgui.h"

namespace {
#include "built_in_shaders.inl"
}

int RunHDR(grassland::rhi::BackendAPI api, int frames, const std::string &capture_directory) {
  using namespace grassland::rhi;

  grassland::rhix::Core core = grassland::rhix::CreateCore(api);
  if (!core)
    return 1;
  grassland::rhix::Device device = core.CreateDeviceByAutoSelection();
  if (!device)
    return 1;
  rhi_demo::LastFrame capture(capture_directory, "hdr");
  grassland::rhix::Queue queue = device.GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue)
    return 1;
  size_t color_space_changes = 0;
  bool present_mode_changed = false;
  PresentMode requested = PRESENT_MODE_VSYNC;
  grassland::rhix::Window window;
  std::vector<ColorSpace> spaces{COLOR_SPACE_SRGB_NONLINEAR};
  window = device.CreateGLFWWindow({960, 540, "RHIX HDR - H: color space, P: toggle VSync, Escape: close"});
  if (!window)
    return 1;
  if (!window.EnableImGui())
    return 1;
  for (const auto &option : window.GetSupportedPresentationOptions()) {
    std::cout << ColorSpaceName(option.color_space) << " / " << ImageFormatName(option.format) << " / "
              << PresentModeName(option.present_mode) << '\n';
    if (std::find(spaces.begin(), spaces.end(), option.color_space) == spaces.end())
      spaces.push_back(option.color_space);
  }
  if (spaces.size() == 1)
    std::cout << "HDR presentation unavailable on this surface; showing SDR explicitly.\n";
  std::cout
      << "hdr: relative linear levels 0.25x, 0.5x, 1x, 2x, 4x reference white; top neutral, bottom Rec.709 primaries.\n";
  const auto pq = std::find(spaces.begin(), spaces.end(), COLOR_SPACE_HDR10_ST2084);
  size_t selected = pq != spaces.end() ? static_cast<size_t>(pq - spaces.begin()) : (spaces.size() > 1 ? 1 : 0);
  if (!window.SetColorSpace(spaces[selected]))
    return 1;
  const auto &embedded = shader_list.at("modules/hdr/shaders/hdr.slang");
  const std::string shader(embedded.first, embedded.second);
  std::map<std::pair<ImageFormat, ColorSpace>, grassland::rhix::Program> programs;
  const BindingLayoutDesc settings_layout{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT}}};
  auto settings_buffer = device.CreateBuffer({256});
  auto settings_group = device.CreateBindingGroup(settings_layout);
  if (!settings_buffer || !settings_group || !settings_group.BindResource(0, settings_buffer))
    return 1;
  float previous_scale = -1.0f;
  float previous_headroom = -1.0f;
  window.KeyEvent().RegisterCallback([&](int key, int, int action, int) {
    if (action != GLFW_PRESS)
      return;
    if (key == GLFW_KEY_ESCAPE)
      glfwSetWindowShouldClose(window.GLFWWindow(), GLFW_TRUE);
    if (key == GLFW_KEY_H)
      ++color_space_changes;
    if (key == GLFW_KEY_P) {
      requested = requested == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
      present_mode_changed = true;
    }
  });
  bool title_dirty = true;
  grassland::FPSCounter fps;
  auto title_time = std::chrono::steady_clock::now();
  int rendered = 0;
  while (!frames || rendered < frames) {
    if (!PollEvents() || window.ShouldClose())
      break;
    if (window.IsSwapChainInvalid() && !window.RebuildSwapChain())
      return 1;
    // Bounded runs exercise all available color spaces automatically.
    const bool cycle = frames >= 3 * static_cast<int>(spaces.size()) && rendered > 0 &&
                       rendered % std::max(1, frames / static_cast<int>(spaces.size())) == 0;
    if (color_space_changes || cycle) {
      selected = (selected + (color_space_changes ? color_space_changes : 1)) % spaces.size();
      color_space_changes = 0;
      if (!window.SetColorSpace(spaces[selected]))
        return 1;
      fps.Reset();
      title_dirty = true;
    }
    if (present_mode_changed) {
      present_mode_changed = false;
      if (!window.SetPresentMode(requested))
        return 1;
      std::cout << "hdr: presentation mode " << PresentModeName(window.GetPresentMode()) << '\n';
      fps.Reset();
      title_dirty = true;
    }
    const float linear_scale = window.GetHDRReferenceWhiteScale();
    const float headroom = window.GetHDRHeadroom();
    if (linear_scale != previous_scale || headroom != previous_headroom) {
      std::cout << "hdr: reference white scale " << linear_scale << "; relative headroom " << headroom << '\n';
      previous_scale = linear_scale;
      previous_headroom = headroom;
    }
    if (!window.UpdateImGui([&] {
          ImGui::Begin("SDR appearance in HDR");
          ImGui::TextUnformatted("H: switch color space; solid UI colors retain their SDR meaning.");
          ImGui::Text("Output: %s", ColorSpaceName(window.GetColorSpace()).c_str());
          ImGui::TextUnformatted("Gray ramp, colored text and translucent controls:");
          for (int i = 0; i < 5; ++i) {
            if (i)
              ImGui::SameLine();
            const float gray = float(i) / 4;
            ImGui::PushID(i);
            ImGui::ColorButton("Gray", {gray, gray, gray, 1}, ImGuiColorEditFlags_NoTooltip, {32, 24});
            ImGui::PopID();
          }
          ImGui::TextColored({1, 0.3f, 0.2f, 1}, "SDR colored text");
          ImGui::End();
        }))
      return 1;
    const float settings[4] = {linear_scale, headroom, 0, 0};
    const auto color_space = window.GetColorSpace();
    grassland::rhix::Image image = window.GetNextImage();
    if (!image) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    const auto extent = image.GetDesc();
    auto &program = programs[{extent.format, color_space}];
    if (!program) {
      const auto source = "#define OUTPUT_COLOR_SPACE " + std::to_string(static_cast<int>(color_space)) + "\n" + shader;
      grassland::rhix::Shader vs = device.CreateShaderFromSource(source, "VSMain", SHADER_STAGE_VERTEX);
      grassland::rhix::Shader ps = device.CreateShaderFromSource(source, "PSMain", SHADER_STAGE_FRAGMENT);
      if (!vs || !ps)
        return 1;
      ProgramDesc desc;
      desc.vertex_shader = vs.Get();
      desc.fragment_shader = ps.Get();
      desc.color_formats = {extent.format};
      desc.pipeline_layout.groups = {settings_layout};
      program = device.CreateProgram(desc);
      if (!program)
        return 1;
      std::cout << "Rendering " << ColorSpaceName(color_space) << " / " << ImageFormatName(extent.format) << '\n';
    }
    grassland::rhix::CommandContext context = device.CreateCommandContext();
    if (!context || !context.CmdUploadBuffer(settings_buffer, 0, settings, sizeof(settings)) ||
        !context.CmdBindProgram(program) || !context.CmdBindGroup(0, settings_group) ||
        !context.CmdBeginRendering(image, ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}) ||
        !context.CmdSetViewport({0, 0, float(extent.width), float(extent.height), 0, 1}) ||
        !context.CmdSetScissor({0, 0, extent.width, extent.height}) || !context.CmdDraw(3) ||
        !context.CmdEndRendering())
      return 1;
    if (!context.CmdPresent(window))
      return 1;
    if (!capture.Record(context.Get(), image.Get(), window.Get()))
      return 1;
    grassland::rhix::SubmissionToken token = queue.Submit(std::move(context));

    if (!token || !token.WaitHost())
      return 1;
    fps.TickFrame();
    // The image and recording leave scope before a mode switch or PollEvents.
    const auto now = std::chrono::steady_clock::now();
    if (title_dirty || now - title_time >= std::chrono::milliseconds(250)) {
      if (!window.SetTitle(fmt::format(
              "RHIX HDR - {} - FPS: {:.1f} - VSync: {} - 0.25x | 0.5x | 1x | 2x | 4x white - H: color space, P: toggle VSync",
              ColorSpaceName(color_space), fps.GetFPS(), window.GetPresentMode() == PRESENT_MODE_VSYNC ? "On" : "Off")))
        return 1;
      title_time = now;
      title_dirty = false;
    }
    ++rendered;
  }
  std::cout << "hdr: completed " << rendered << " frames (window presentation).\n";
  if (!capture.Save())
    return 1;
  return 0;
}
