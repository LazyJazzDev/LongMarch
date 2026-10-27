#include "module.h"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <map>
#include <thread>

#include "../../save_image.h"
#include "grassland/rhi/rhix/rhix.h"
#include "grassland/util/fps_counter.h"
#include "grassland/util/virtual_file_system.h"
#include "imgui.h"

namespace {
#include "built_in_shaders.inl"
}

using namespace grassland::rhi;
using namespace grassland::rhix;

struct HDRModule::State {
  bool initialized{};
  int frames{};
  std::string capture_directory{};
  grassland::rhix::Device device{};
  grassland::rhix::DownloadToken last_frame{};
  ImageDesc last_frame_desc{};
  ColorSpace last_frame_space{};
  float last_frame_white_scale{};
  grassland::rhix::Queue queue{};
  size_t color_space_changes{};
  bool present_mode_changed{};
  PresentMode requested{};
  grassland::rhix::Window window{};
  std::vector<ColorSpace> spaces{};
  size_t selected{};
  std::string shader{};
  std::map<std::pair<ImageFormat, ColorSpace>, grassland::rhix::Program> programs{};
  BindingLayoutDesc settings_layout{};
  Buffer settings_buffer{};
  BindingGroup settings_group{};
  float previous_scale{};
  float previous_headroom{};
  bool title_dirty{};
  grassland::FPSCounter fps{};
  std::chrono::steady_clock::time_point title_time{};
  std::array<float, 4> settings{};
  grassland::rhi::ColorSpace color_space{};
};

HDRModule::HDRModule(grassland::rhix::Core core, grassland::rhix::Device device)
    : ModuleBase(std::move(core), std::move(device)) {
}

HDRModule::~HDRModule() = default;

int HDRModule::OnInit(const ModuleOptions &options) {
  state_ = std::make_unique<State>();

  state_->frames = options.frames;
  state_->capture_directory = options.capture_directory;
  using namespace grassland::rhi;
  using namespace grassland::rhix;
  state_->device = GetDevice();

  state_->last_frame_desc = ImageDesc{};
  state_->last_frame_space = COLOR_SPACE_SRGB_NONLINEAR;
  state_->last_frame_white_scale = 1.0f;
  state_->queue = state_->device.GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!state_->queue)
    return 1;
  state_->color_space_changes = 0;
  state_->present_mode_changed = false;
  state_->requested = PRESENT_MODE_VSYNC;

  state_->spaces = std::vector<ColorSpace>{COLOR_SPACE_SRGB_NONLINEAR};
  state_->window =
      state_->device.CreateGLFWWindow({960, 540, "RHIX HDR - H: color space, P: toggle VSync, Escape: close"});
  if (!state_->window)
    return 1;
  if (!state_->window.EnableImGui())
    return 1;
  for (const auto &option : state_->window.GetSupportedPresentationOptions()) {
    std::cout << ColorSpaceName(option.color_space) << " / " << ImageFormatName(option.format) << " / "
              << PresentModeName(option.present_mode) << '\n';
    if (std::find(state_->spaces.begin(), state_->spaces.end(), option.color_space) == state_->spaces.end())
      state_->spaces.push_back(option.color_space);
  }
  if (state_->spaces.size() == 1)
    std::cout << "HDR presentation unavailable on this surface; showing SDR explicitly.\n";
  std::cout
      << "hdr: relative linear levels 0.25x, 0.5x, 1x, 2x, 4x reference white; top neutral, bottom Rec.709 primaries.\n";
  const auto pq = std::find(state_->spaces.begin(), state_->spaces.end(), COLOR_SPACE_HDR10_ST2084);
  state_->selected = pq != state_->spaces.end() ? static_cast<size_t>(pq - state_->spaces.begin())
                                                : (state_->spaces.size() > 1 ? 1 : 0);
  if (!state_->window.SetColorSpace(state_->spaces[state_->selected]))
    return 1;
  const auto &embedded = shader_list.at("modules/hdr/shaders/hdr.slang");
  state_->shader = std::string(embedded.first, embedded.second);

  state_->settings_layout = BindingLayoutDesc{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT}}};
  state_->settings_buffer = state_->device.CreateBuffer({256});
  state_->settings_group = state_->device.CreateBindingGroup(state_->settings_layout);
  if (!state_->settings_buffer || !state_->settings_group ||
      !state_->settings_group.BindResource(0, state_->settings_buffer))
    return 1;
  state_->previous_scale = -1.0f;
  state_->previous_headroom = -1.0f;
  state_->window.KeyEvent().RegisterCallback([this](int key, int, int action, int) {
    if (action != GLFW_PRESS)
      return;
    if (key == GLFW_KEY_ESCAPE)
      glfwSetWindowShouldClose(state_->window.GLFWWindow(), GLFW_TRUE);
    if (key == GLFW_KEY_H)
      ++state_->color_space_changes;
    if (key == GLFW_KEY_P) {
      state_->requested = state_->requested == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
      state_->present_mode_changed = true;
    }
  });
  state_->title_dirty = true;

  state_->title_time = std::chrono::steady_clock::now();
  SetWarmupFrames(0);

  state_->initialized = true;
  return 0;
}

int HDRModule::OnUpdate() {
  if (state_->window.ShouldClose()) {
    RequestClose();
    return 0;
  }
  if (state_->window.IsSwapChainInvalid() && !state_->window.RebuildSwapChain())
    return 1;
  // Bounded runs exercise all available color spaces automatically.
  const bool cycle = state_->frames >= 3 * static_cast<int>(state_->spaces.size()) && GetFrameIndex() > 0 &&
                     GetFrameIndex() % std::max(1, state_->frames / static_cast<int>(state_->spaces.size())) == 0;
  if (state_->color_space_changes || cycle) {
    state_->selected =
        (state_->selected + (state_->color_space_changes ? state_->color_space_changes : 1)) % state_->spaces.size();
    state_->color_space_changes = 0;
    if (!state_->window.SetColorSpace(state_->spaces[state_->selected]))
      return 1;
    state_->fps.Reset();
    state_->title_dirty = true;
  }
  if (state_->present_mode_changed) {
    state_->present_mode_changed = false;
    if (!state_->window.SetPresentMode(state_->requested))
      return 1;
    std::cout << "hdr: presentation mode " << PresentModeName(state_->window.GetPresentMode()) << '\n';
    state_->fps.Reset();
    state_->title_dirty = true;
  }
  const float linear_scale = state_->window.GetHDRReferenceWhiteScale();
  const float headroom = state_->window.GetHDRHeadroom();
  if (linear_scale != state_->previous_scale || headroom != state_->previous_headroom) {
    std::cout << "hdr: reference white scale " << linear_scale << "; relative headroom " << headroom << '\n';
    state_->previous_scale = linear_scale;
    state_->previous_headroom = headroom;
  }
  if (!state_->window.UpdateImGui([&] {
        ImGui::Begin("SDR appearance in HDR");
        ImGui::TextUnformatted("H: switch color space; solid UI colors retain their SDR meaning.");
        ImGui::Text("Output: %s", ColorSpaceName(state_->window.GetColorSpace()).c_str());
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
  state_->settings = {linear_scale, headroom, 0, 0};
  state_->color_space = state_->window.GetColorSpace();

  return 0;
}

int HDRModule::OnRender() {
  grassland::rhix::Image image = state_->window.GetNextImage();
  if (!image) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return 0;
  }
  const auto extent = image.GetDesc();
  auto &program = state_->programs[{extent.format, state_->color_space}];
  if (!program) {
    const auto source =
        "#define OUTPUT_COLOR_SPACE " + std::to_string(static_cast<int>(state_->color_space)) + "\n" + state_->shader;
    grassland::rhix::Shader vs = state_->device.CreateShaderFromSource(source, "VSMain", SHADER_STAGE_VERTEX);
    grassland::rhix::Shader ps = state_->device.CreateShaderFromSource(source, "PSMain", SHADER_STAGE_FRAGMENT);
    if (!vs || !ps)
      return 1;
    ProgramDesc desc;
    desc.vertex_shader = vs.Get();
    desc.fragment_shader = ps.Get();
    desc.color_formats = {extent.format};
    desc.pipeline_layout.groups = {state_->settings_layout};
    program = state_->device.CreateProgram(desc);
    if (!program)
      return 1;
    std::cout << "Rendering " << ColorSpaceName(state_->color_space) << " / " << ImageFormatName(extent.format) << '\n';
  }
  grassland::rhix::CommandContext context = state_->device.CreateCommandContext();
  if (!context ||
      !context.CmdUploadBuffer(state_->settings_buffer, 0, state_->settings.data(), sizeof(state_->settings)) ||
      !context.CmdBindProgram(program) || !context.CmdBindGroup(0, state_->settings_group) ||
      !context.CmdBeginRendering(image, ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}) ||
      !context.CmdSetViewport({0, 0, float(extent.width), float(extent.height), 0, 1}) ||
      !context.CmdSetScissor({0, 0, extent.width, extent.height}) || !context.CmdDraw(3) || !context.CmdEndRendering())
    return 1;
  if (!context.CmdPresent(state_->window))
    return 1;
  if (!state_->capture_directory.empty()) {
    state_->last_frame = context.CmdDownloadImage(image);
    if (!state_->last_frame)
      return 1;
    state_->last_frame_desc = image.GetDesc();
    state_->last_frame_space = state_->window ? state_->window.GetColorSpace() : COLOR_SPACE_SRGB_NONLINEAR;
    state_->last_frame_white_scale = state_->window ? state_->window.GetHDRReferenceWhiteScale() : 1.0f;
  }
  grassland::rhix::SubmissionToken token = state_->queue.Submit(std::move(context));

  if (!token || !token.WaitHost())
    return 1;
  state_->fps.TickFrame();
  // The image and recording leave scope before a mode switch or PollEvents.
  const auto now = std::chrono::steady_clock::now();
  if (state_->title_dirty || now - state_->title_time >= std::chrono::milliseconds(250)) {
    if (!state_->window.SetTitle(fmt::format(
            "RHIX HDR - {} - FPS: {:.1f} - VSync: {} - 0.25x | 0.5x | 1x | 2x | 4x white - H: color space, P: toggle VSync",
            ColorSpaceName(state_->color_space), state_->fps.GetFPS(),
            state_->window.GetPresentMode() == PRESENT_MODE_VSYNC ? "On" : "Off")))
      return 1;
    state_->title_time = now;
    state_->title_dirty = false;
  }
  CompleteFrame();

  return 0;
}

int HDRModule::OnTerminate() {
  int result = 0;
  if (state_ && state_->initialized) {
    result = [&]() -> int {
      std::cout << "hdr: completed " << GetFrameIndex() << " frames (window presentation).\n";
      if (!state_->capture_directory.empty()) {
        if (!state_->last_frame || !state_->last_frame.WaitHost())
          return 1;
        std::vector<uint8_t> pixels(state_->last_frame.GetSize());
        if (!state_->last_frame.CopyTo(pixels.data(), pixels.size()) ||
            !SaveImage(state_->capture_directory, "hdr", pixels, state_->last_frame_desc, state_->last_frame_space,
                       state_->last_frame_white_scale))
          return 1;
      }
      return 0;
    }();
  }
  state_.reset();
  return result;
}
