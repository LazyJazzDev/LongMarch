#include "module.h"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <map>
#include <thread>

#include "../../save_image.h"
#include "grassland/rhi/rhi.h"
#include "grassland/util/fps_counter.h"
#include "grassland/util/virtual_file_system.h"
#include "imgui.h"

namespace {
#include "built_in_shaders.inl"
}

using namespace grassland::rhi;

int HDRModule::OnInit(const ModuleOptions &options) {
  frames_ = options.frames;
  capture_directory_ = options.capture_directory;
  using namespace grassland::rhi;
  device_ = GetDevice();

  last_frame_desc_ = ImageDesc{};
  last_frame_space_ = COLOR_SPACE_SRGB_NONLINEAR;
  last_frame_white_scale_ = 1.0f;
  queue_ = device_->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue_)
    return 1;
  color_space_changes_ = 0;
  present_mode_changed_ = false;
  requested_ = PRESENT_MODE_VSYNC;

  spaces_ = std::vector<ColorSpace>{COLOR_SPACE_SRGB_NONLINEAR};
  window_ = device_->CreateGLFWWindow({960, 540, "RHI HDR - H: color space, P: toggle VSync, Escape: close"});
  if (!window_)
    return 1;
  if (!window_->EnableImGui())
    return 1;
  for (const auto &option : window_->GetSupportedPresentationOptions()) {
    std::cout << ColorSpaceName(option.color_space) << " / " << ImageFormatName(option.format) << " / "
              << PresentModeName(option.present_mode) << '\n';
    if (std::find(spaces_.begin(), spaces_.end(), option.color_space) == spaces_.end())
      spaces_.push_back(option.color_space);
  }
  if (spaces_.size() == 1)
    std::cout << "HDR presentation unavailable on this surface; showing SDR explicitly.\n";
  std::cout
      << "hdr: relative linear levels 0.25x, 0.5x, 1x, 2x, 4x reference white; top neutral, bottom Rec.709 primaries.\n";
  const auto pq = std::find(spaces_.begin(), spaces_.end(), COLOR_SPACE_HDR10_ST2084);
  selected_ = pq != spaces_.end() ? static_cast<size_t>(pq - spaces_.begin()) : (spaces_.size() > 1 ? 1 : 0);
  if (!window_->SetColorSpace(spaces_[selected_]))
    return 1;
  const auto &embedded = shader_list.at("modules/hdr/shaders/hdr.slang");
  shader_ = std::string(embedded.first, embedded.second);

  settings_layout_ = BindingLayoutDesc{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT}}};
  settings_buffer_ = device_->CreateBuffer({256});
  settings_group_ = device_->CreateBindingGroup(settings_layout_);
  if (!settings_buffer_ || !settings_group_ || !settings_group_->BindResource(0, settings_buffer_.Get()))
    return 1;
  previous_scale_ = -1.0f;
  previous_headroom_ = -1.0f;
  window_->KeyEvent().RegisterCallback([this](int key, int, int action, int) {
    if (action != GLFW_PRESS)
      return;
    if (key == GLFW_KEY_ESCAPE)
      glfwSetWindowShouldClose(window_->GLFWWindow(), GLFW_TRUE);
    if (key == GLFW_KEY_H)
      ++color_space_changes_;
    if (key == GLFW_KEY_P) {
      requested_ = requested_ == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
      present_mode_changed_ = true;
    }
  });
  title_dirty_ = true;

  title_time_ = std::chrono::steady_clock::now();
  SetWarmupFrames(0);

  initialized_ = true;
  return 0;
}

int HDRModule::OnUpdate() {
  if (window_->ShouldClose()) {
    RequestClose();
    return 0;
  }
  if (window_->IsSwapChainInvalid() && !window_->RebuildSwapChain())
    return 1;
  // Bounded runs exercise all available color spaces automatically.
  const bool cycle = frames_ >= 3 * static_cast<int>(spaces_.size()) && GetFrameIndex() > 0 &&
                     GetFrameIndex() % std::max(1, frames_ / static_cast<int>(spaces_.size())) == 0;
  if (color_space_changes_ || cycle) {
    selected_ = (selected_ + (color_space_changes_ ? color_space_changes_ : 1)) % spaces_.size();
    color_space_changes_ = 0;
    if (!window_->SetColorSpace(spaces_[selected_]))
      return 1;
    fps_.Reset();
    title_dirty_ = true;
  }
  if (present_mode_changed_) {
    present_mode_changed_ = false;
    if (!window_->SetPresentMode(requested_))
      return 1;
    std::cout << "hdr: presentation mode " << PresentModeName(window_->GetPresentMode()) << '\n';
    fps_.Reset();
    title_dirty_ = true;
  }
  const float linear_scale = window_->GetHDRReferenceWhiteScale();
  const float headroom = window_->GetHDRHeadroom();
  if (linear_scale != previous_scale_ || headroom != previous_headroom_) {
    std::cout << "hdr: reference white scale " << linear_scale << "; relative headroom " << headroom << '\n';
    previous_scale_ = linear_scale;
    previous_headroom_ = headroom;
  }
  if (!window_->UpdateImGui([&] {
        ImGui::Begin("SDR appearance in HDR");
        ImGui::TextUnformatted("H: switch color space; solid UI colors retain their SDR meaning.");
        ImGui::Text("Output: %s", ColorSpaceName(window_->GetColorSpace()).c_str());
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
  settings_ = {linear_scale, headroom, 0, 0};
  color_space_ = window_->GetColorSpace();

  return 0;
}

int HDRModule::OnRender() {
  ObjPtr<IImage> image = window_->GetNextImage();
  if (!image) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return 0;
  }
  const auto extent = image->GetDesc();
  auto &program = programs_[{extent.format, color_space_}];
  if (!program) {
    const auto source = "#define OUTPUT_COLOR_SPACE " + std::to_string(static_cast<int>(color_space_)) + "\n" + shader_;
    ObjPtr<IShader> vs = device_->CreateShaderFromSource(source, "VSMain", SHADER_STAGE_VERTEX);
    ObjPtr<IShader> ps = device_->CreateShaderFromSource(source, "PSMain", SHADER_STAGE_FRAGMENT);
    if (!vs || !ps)
      return 1;
    ProgramDesc desc;
    desc.vertex_shader = vs.Get();
    desc.fragment_shader = ps.Get();
    desc.color_formats = {extent.format};
    desc.pipeline_layout.groups = {settings_layout_};
    program = device_->CreateProgram(desc);
    if (!program)
      return 1;
    std::cout << "Rendering " << ColorSpaceName(color_space_) << " / " << ImageFormatName(extent.format) << '\n';
  }
  ObjPtr<ICommandContext> context = device_->CreateCommandContext();
  if (!context || !context->CmdUploadBuffer(settings_buffer_.Get(), 0, settings_.data(), sizeof(settings_)) ||
      !context->CmdBindProgram(program.Get()) || !context->CmdBindGroup(0, settings_group_.Get()) ||
      !context->CmdBeginRendering(
          {{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}) ||
      !context->CmdSetViewport({0, 0, float(extent.width), float(extent.height), 0, 1}) ||
      !context->CmdSetScissor({0, 0, extent.width, extent.height}) || !context->CmdDraw(3) ||
      !context->CmdEndRendering())
    return 1;
  if (!context->CmdPresent(window_.Get()))
    return 1;
  if (!capture_directory_.empty()) {
    last_frame_ = context->CmdDownloadImage(image.Get());
    if (!last_frame_)
      return 1;
    last_frame_desc_ = image->GetDesc();
    last_frame_space_ = window_ ? window_->GetColorSpace() : COLOR_SPACE_SRGB_NONLINEAR;
    last_frame_white_scale_ = window_ ? window_->GetHDRReferenceWhiteScale() : 1.0f;
  }
  ObjPtr<ISubmissionToken> token = queue_->Submit(context.Get());

  if (!token || !token->WaitHost())
    return 1;
  fps_.TickFrame();
  // The image and recording leave scope before a mode switch or PollEvents.
  const auto now = std::chrono::steady_clock::now();
  if (title_dirty_ || now - title_time_ >= std::chrono::milliseconds(250)) {
    if (!window_->SetTitle(fmt::format(
            "RHI HDR - {} - FPS: {:.1f} - VSync: {} - 0.25x | 0.5x | 1x | 2x | 4x white - H: color space, P: toggle VSync",
            ColorSpaceName(color_space_), fps_.GetFPS(),
            window_->GetPresentMode() == PRESENT_MODE_VSYNC ? "On" : "Off")))
      return 1;
    title_time_ = now;
    title_dirty_ = false;
  }
  CompleteFrame();

  return 0;
}

int HDRModule::OnTerminate() {
  int result = 0;
  if (initialized_) {
    result = [&]() -> int {
      std::cout << "hdr: completed " << GetFrameIndex() << " frames (window presentation).\n";
      if (!capture_directory_.empty()) {
        if (!last_frame_ || !last_frame_->WaitHost())
          return 1;
        std::vector<uint8_t> pixels(last_frame_->GetSize());
        if (!last_frame_->CopyTo(pixels.data(), pixels.size()) ||
            !SaveImage(capture_directory_, "hdr", pixels, last_frame_desc_, last_frame_space_, last_frame_white_scale_))
          return 1;
      }
      return 0;
    }();
  }
  color_space_ = {};
  settings_ = {};
  title_time_ = {};
  fps_ = {};
  title_dirty_ = {};
  previous_headroom_ = {};
  previous_scale_ = {};
  settings_group_ = {};
  settings_buffer_ = {};
  settings_layout_ = {};
  programs_ = {};
  shader_ = {};
  selected_ = {};
  spaces_ = {};
  window_ = {};
  requested_ = {};
  present_mode_changed_ = {};
  color_space_changes_ = {};
  queue_ = {};
  last_frame_white_scale_ = {};
  last_frame_space_ = {};
  last_frame_desc_ = {};
  last_frame_ = {};
  device_ = {};
  capture_directory_ = {};
  frames_ = {};
  initialized_ = {};
  return result;
}
