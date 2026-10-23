#include "../../module.h"

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <thread>

#include "grassland/rhi/rhix/rhix.h"
#include "grassland/util/fps_counter.h"
#include "grassland/util/virtual_file_system.h"

namespace {
#include "built_in_shaders.inl"
}

int RunHDR(grassland::rhi::BackendAPI api, int frames, bool headless) {
  using namespace grassland::rhi;

  grassland::rhix::Core core = grassland::rhix::CreateCore(api);
  if (!core)
    return 1;
  grassland::rhix::Device device = core.CreateDeviceByAutoSelection();
  if (!device)
    return 1;
  grassland::rhix::Queue queue = device.GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue)
    return 1;
  grassland::rhix::Window window;
  std::vector<ColorSpace> spaces{COLOR_SPACE_SRGB_NONLINEAR};
  if (!headless) {
    window = device.CreateGLFWWindow({960, 540, "RHIX HDR - H: color space, P: toggle VSync, Escape: close"});
    if (!window)
      return 1;
    for (const auto &option : window.GetSupportedPresentationOptions()) {
      std::cout << ColorSpaceName(option.color_space) << " / " << ImageFormatName(option.format) << " / "
                << PresentModeName(option.present_mode) << '\n';
      if (std::find(spaces.begin(), spaces.end(), option.color_space) == spaces.end())
        spaces.push_back(option.color_space);
    }
    if (spaces.size() == 1)
      std::cout << "HDR presentation unavailable on this surface; showing SDR explicitly.\n";
  } else {
    spaces = {COLOR_SPACE_SRGB_NONLINEAR, COLOR_SPACE_EXTENDED_SRGB_LINEAR, COLOR_SPACE_HDR10_ST2084,
              COLOR_SPACE_BT709_LINEAR};
  }
  std::cout
      << "hdr: relative linear levels 0.25x, 0.5x, 1x, 2x, 4x reference white; top neutral, bottom Rec.709 primaries.\n";
  const auto pq = std::find(spaces.begin(), spaces.end(), COLOR_SPACE_HDR10_ST2084);
  size_t selected = pq != spaces.end() ? static_cast<size_t>(pq - spaces.begin()) : (spaces.size() > 1 ? 1 : 0);
  if (window && !window.SetColorSpace(spaces[selected]))
    return 1;
  const auto &embedded = shader_list.at("modules/hdr/shaders/hdr.slang");
  const std::string shader(embedded.first, embedded.second);
  std::map<std::pair<ImageFormat, ColorSpace>, grassland::rhix::Program> programs;
  std::map<ColorSpace, grassland::rhix::Image> targets;
  const BindingLayoutDesc settings_layout{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT}}};
  auto settings_buffer = device.CreateBuffer({256});
  auto settings_group = device.CreateBindingGroup(settings_layout);
  if (!settings_buffer || !settings_group || !settings_group.BindResource(0, settings_buffer))
    return 1;
  float previous_scale = -1.0f;
  float previous_headroom = -1.0f;
  bool previous_h = false;
  bool previous_p = false;
  bool title_dirty = true;
  grassland::FPSCounter fps;
  auto title_time = std::chrono::steady_clock::now();
  int rendered = 0;
  while (!frames || rendered < frames) {
    if (!headless) {
      if (!PollEvents() || window.ShouldClose() || window.IsKeyDown(GLFW_KEY_ESCAPE))
        break;
      if (window.IsSwapChainInvalid() && !window.RebuildSwapChain())
        return 1;
      const bool h = window.IsKeyDown(GLFW_KEY_H);
      // Bounded runs exercise all available color spaces automatically.
      const bool cycle = frames >= 3 * static_cast<int>(spaces.size()) && rendered > 0 &&
                         rendered % std::max(1, frames / static_cast<int>(spaces.size())) == 0;
      if ((h && !previous_h) || cycle) {
        selected = (selected + 1) % spaces.size();
        if (!window.SetColorSpace(spaces[selected]))
          return 1;
        fps.Reset();
        title_dirty = true;
      }
      previous_h = h;
      const bool p = window.IsKeyDown(GLFW_KEY_P);
      if (p && !previous_p) {
        const auto requested =
            window.GetPresentMode() == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
        if (!window.SetPresentMode(requested))
          return 1;
        std::cout << "hdr: presentation mode " << PresentModeName(window.GetPresentMode()) << '\n';
        fps.Reset();
        title_dirty = true;
      }
      previous_p = p;
    } else {
      selected = (rendered / std::max(1, frames / static_cast<int>(spaces.size()))) % spaces.size();
    }
    const float linear_scale = window ? window.GetHDRReferenceWhiteScale() : 1.0f;
    const float headroom = window ? window.GetHDRHeadroom() : 10.0f;
    if (linear_scale != previous_scale || headroom != previous_headroom) {
      std::cout << "hdr: reference white scale " << linear_scale << "; relative headroom " << headroom << '\n';
      previous_scale = linear_scale;
      previous_headroom = headroom;
    }
    const float settings[4] = {linear_scale, headroom, 0, 0};
    const auto color_space = headless ? spaces[selected] : window.GetColorSpace();
    grassland::rhix::Image image;
    if (headless) {
      auto &target = targets[color_space];
      if (!target) {
        const auto format = (color_space == COLOR_SPACE_EXTENDED_SRGB_LINEAR || color_space == COLOR_SPACE_BT709_LINEAR)
                                ? IMAGE_FORMAT_RGBA16_FLOAT
                            : color_space == COLOR_SPACE_HDR10_ST2084 ? IMAGE_FORMAT_RGB10A2_UNORM
                                                                      : IMAGE_FORMAT_RGBA8_UNORM;
        target = device.CreateImage({640, 360, format});
        if (!target)
          return 1;
      }
      image = target;
    } else
      image = window.GetNextImage();
    if (!image) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    const auto extent = image.GetDesc();
    auto &program = programs[{extent.format, color_space}];
    if (!program) {
      const auto source = "#define OUTPUT_COLOR_SPACE " + std::to_string(static_cast<int>(color_space)) + "\n" + shader;
      grassland::rhix::Shader vs = device.CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
      grassland::rhix::Shader ps = device.CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
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
    if (!headless && !context.CmdPresent(window))
      return 1;
    grassland::rhix::SubmissionToken token = queue.Submit(std::move(context));
    if (!token || !token.WaitHost())
      return 1;
    fps.TickFrame();
    // The image and recording leave scope before a mode switch or PollEvents.
    const auto now = std::chrono::steady_clock::now();
    if (!headless && (title_dirty || now - title_time >= std::chrono::milliseconds(250))) {
      if (!window.SetTitle(fmt::format(
              "RHIX HDR - {} - FPS: {:.1f} - VSync: {} - 0.25x | 0.5x | 1x | 2x | 4x white - H: color space, P: toggle VSync",
              ColorSpaceName(color_space), fps.GetFPS(), window.GetPresentMode() == PRESENT_MODE_VSYNC ? "On" : "Off")))
        return 1;
      title_time = now;
      title_dirty = false;
    }
    ++rendered;
  }
  std::cout << "hdr: completed " << rendered << " frames ("
            << (headless ? "offscreen encoding; no HDR display validation" : "window presentation") << ").\n";
  return 0;
}
