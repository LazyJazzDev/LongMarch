#include "module.h"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <random>
#include <string>
#include <thread>

#include "../../save_image.h"
#include "grassland/rhi/rhi.h"
#include "grassland/util/fps_counter.h"
#include "grassland/util/virtual_file_system.h"
#include "imgui.h"

namespace {
#include "built_in_shaders.inl"

constexpr uint32_t particle_count = 65536;
// Same algorithmic estimate as nbody_cs: 20 FLOP per particle interaction.
// The rate uses whole-frame time, including copies and particle rendering.
constexpr uint64_t flops_per_frame = uint64_t{20} * particle_count * particle_count;

struct Simulation {
  uint32_t count;
  float delta_t, gravity;
};

struct Camera {
  glm::mat4 world_to_screen, camera_to_world;
  float particle_size;
};
}  // namespace

using namespace grassland::rhi;

int NBodyModule::OnInit(const ModuleOptions &options) {
  headless_ = options.headless;
  capture_directory_ = options.capture_directory;
  api_ = GetBackendAPI();
  using namespace grassland::rhi;
  device_ = GetDevice();

  last_frame_desc_ = ImageDesc{};
  last_frame_space_ = COLOR_SPACE_SRGB_NONLINEAR;
  last_frame_white_scale_ = 1.0f;
  requested_ = PRESENT_MODE_VSYNC;
  present_mode_changed_ = false;
  running_ = true;
  reset_requested_ = true;
  delta_t_ = 0.03f;
  galaxy_count_ = 10;
  rotation_ = glm::mat4{1.0f};
  last_cursor_ = glm::dvec2{};
  rotation_delta_ = glm::dvec2{};
  cursor_initialized_ = false;
  dragging_ = false;

  capture_keyboard_ = false;
  output_change_requested_ = false;
  show_gui_ = true;
  requested_space_ = COLOR_SPACE_SRGB_NONLINEAR;
  spaces_ = std::vector<ColorSpace>{COLOR_SPACE_SRGB_NONLINEAR};

  if (!headless_) {
    window_ = device_->CreateGLFWWindow({1280, 720, "RHI nbody"});
    if (!window_)
      return 1;
    if (!window_->EnableImGui(true, LONGMARCH_ASSETS_DIR "fonts/simhei.ttf", 20.0f))
      return 1;
    for (const auto &option : window_->GetSupportedPresentationOptions()) {
      if (std::find(spaces_.begin(), spaces_.end(), option.color_space) == spaces_.end())
        spaces_.push_back(option.color_space);
    }
  }
  queue_ = device_->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue_)
    return 1;

  if (headless_) {
    for (auto &target : targets_) {
      target = device_->CreateImage({1280, 720, IMAGE_FORMAT_R32G32B32A32_SFLOAT});
      if (!target)
        return 1;
    }
  }
  const auto &embedded = shader_list.at("modules/nbody/shaders/particle.slang");
  std::string source(embedded.first, embedded.second);
  ObjPtr<IShader> vertex_shader = device_->CreateShaderFromSource(source, "VSMain", SHADER_STAGE_VERTEX);
  if (!vertex_shader)
    return 1;
  ObjPtr<IShader> fragment_shader = device_->CreateShaderFromSource(source, "PSMain", SHADER_STAGE_FRAGMENT);
  if (!fragment_shader)
    return 1;
  ProgramDesc desc;
  desc.vertex_shader = vertex_shader.Get();
  desc.fragment_shader = fragment_shader.Get();
  desc.blend_states = {BlendState(BLEND_FACTOR_ONE, BLEND_FACTOR_ONE, BLEND_OP_ADD, BLEND_FACTOR_ONE,
                                  BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, BLEND_OP_ADD)};
  desc.vertex_bindings = {{sizeof(glm::vec3), true}};
  desc.vertex_attributes = {{0, 0, INPUT_TYPE_FLOAT3, 0}};
  desc.pipeline_layout.groups = {{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_VERTEX}}}};
  desc.color_formats = {IMAGE_FORMAT_R32G32B32A32_SFLOAT};
  program_ = device_->CreateProgram(desc);
  if (!program_)
    return 1;

  const auto &output_embedded = shader_list.at("modules/nbody/shaders/present.slang");
  output_source_ = std::string(output_embedded.first, output_embedded.second);
  output_layout_ = BindingLayoutDesc{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT},
                                      {1, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT}}};
  output_uniform_ = device_->CreateBuffer({256});
  output_group_ = device_->CreateBindingGroup(output_layout_);
  if (!output_uniform_ || !output_group_ || !output_group_->BindResource(0, output_uniform_.Get()))
    return 1;

  const auto &compute_source = shader_list.at("modules/nbody/shaders/nbody.slang");
  ObjPtr<IShader> compute_shader = device_->CreateShaderFromSource(
      std::string(compute_source.first, compute_source.second), "CSMain", SHADER_STAGE_COMPUTE);
  if (!compute_shader)
    return 1;
  ComputeProgramDesc compute_desc;
  compute_desc.compute_shader = compute_shader.Get();
  compute_desc.pipeline_layout.groups = {{{{0, BINDING_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                                           {1, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                                           {2, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                                           {3, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_COMPUTE}}}};
  compute_ = device_->CreateComputeProgram(compute_desc);
  if (!compute_)
    return 1;
  bytes_ = particle_count * sizeof(glm::vec3);
  positions_ = device_->CreateBuffer({bytes_});
  if (!positions_)
    return 1;
  velocities_ = device_->CreateBuffer({bytes_});
  if (!velocities_)
    return 1;
  next_positions_ = device_->CreateBuffer({bytes_});
  if (!next_positions_)
    return 1;
  simulation_uniform_ = device_->CreateBuffer({256});
  if (!simulation_uniform_)
    return 1;
  camera_uniform_ = device_->CreateBuffer({256});
  if (!camera_uniform_)
    return 1;
  simulation_group_ = device_->CreateBindingGroup(compute_desc.pipeline_layout.groups[0]);
  if (!simulation_group_)
    return 1;
  if (!simulation_group_->BindResource(0, positions_.Get()))
    return 1;
  if (!simulation_group_->BindResource(1, velocities_.Get()))
    return 1;
  if (!simulation_group_->BindResource(2, next_positions_.Get()))
    return 1;
  if (!simulation_group_->BindResource(3, simulation_uniform_.Get()))
    return 1;
  camera_group_ = device_->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  if (!camera_group_)
    return 1;
  if (!camera_group_->BindResource(0, camera_uniform_.Get()))
    return 1;
  random_ = std::mt19937(42);

  if (!headless_) {
    window_->KeyEvent().RegisterCallback([this](int key, int, int action, int) {
      if (action != GLFW_PRESS)
        return;
      if (key == GLFW_KEY_I) {
        show_gui_ = !show_gui_;
        capture_keyboard_ = false;
        return;
      }
      if (capture_keyboard_ && key != GLFW_KEY_ESCAPE)
        return;
      if (key == GLFW_KEY_SPACE)
        running_ = !running_;
      if (key == GLFW_KEY_R)
        reset_requested_ = true;
      if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window_->GLFWWindow(), GLFW_TRUE);
      if (key == GLFW_KEY_P) {
        requested_ = requested_ == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
        present_mode_changed_ = true;
      }
    });
  }
  if (window_) {
    window_->MouseButtonEvent().RegisterCallback([this](int button, int action, int, double, double) {
      if (button == GLFW_MOUSE_BUTTON_LEFT)
        dragging_ = action != GLFW_RELEASE;
    });
    window_->MouseMoveEvent().RegisterCallback([this](double x, double y) {
      const glm::dvec2 cursor{x, y};
      if (cursor_initialized_ && dragging_)
        rotation_delta_ += cursor - last_cursor_;
      last_cursor_ = cursor;
      cursor_initialized_ = true;
    });
  }

  downloaded_velocities_ = std::vector<glm::vec3>(particle_count);
  speed_counts_ = std::array<uint32_t, 100>{};
  normalized_speeds_ = std::array<float, 100>{};
  maximum_speed_ = 0;
  speeds_ready_ = false;

  // Warm up allocations and pipelines before measuring completed GPU work.
  benchmark_start_ = std::chrono::steady_clock::now();
  SetWarmupFrames(headless_ ? 60 : 0);

  initialized_ = true;
  return 0;
}

int NBodyModule::OnUpdate() {
  if (!UpdateSpeeds())
    return 1;
  if (!headless_) {
    if (window_->ShouldClose()) {
      RequestClose();
      return 0;
    }
    if (!window_->UpdateImGui([&] {
          if (show_gui_) {
            ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Once);
            ImGui::SetNextWindowBgAlpha(0.9f);
            ImGui::SetNextWindowSize({ImGui::GetFontSize() * 30.0f, 0}, ImGuiCond_Always);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {8.0f, 6.0f});
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
            if (ImGui::Begin(
                    "NBody", nullptr,
                    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize)) {
              ImGui::SeparatorText("Statistics");
              ImGui::Text("Backend: %s", BackendAPIName(api_).c_str());
              ImGui::Text("Particles: %u", particle_count);
              const double rate = fps_.GetFPS();
              ImGui::Text("FPS: %.1f", rate);
              ImGui::Text("Frame Duration: %.3f ms", rate > 0 ? 1000.0 / rate : 0.0);
              ImGui::Text("Estimated: %.2f TFLOP/s", running_ ? flops_per_frame * rate * 1e-12 : 0.0);
              ImGui::SeparatorText("Control");
              const auto &style = ImGui::GetStyle();
              const float row_width = ImGui::GetContentRegionAvail().x;
              const ImVec2 button_size{(row_width - style.ItemSpacing.x) * 0.5f, ImGui::GetFrameHeight() + 6.0f};
              ImGui::PushStyleColor(ImGuiCol_Button, {0.20f, 0.25f, 0.32f, 1.0f});
              ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.28f, 0.35f, 0.44f, 1.0f});
              ImGui::PushStyleColor(ImGuiCol_ButtonActive, {0.33f, 0.41f, 0.52f, 1.0f});
              if (ImGui::Button("Reset", button_size))
                reset_requested_ = true;
              ImGui::PopStyleColor(3);
              ImGui::SameLine();
              ImGui::PushStyleColor(ImGuiCol_Button,
                                    running_ ? ImVec4{0.16f, 0.34f, 0.52f, 1.0f} : ImVec4{0.14f, 0.38f, 0.28f, 1.0f});
              ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                    running_ ? ImVec4{0.20f, 0.43f, 0.65f, 1.0f} : ImVec4{0.18f, 0.49f, 0.36f, 1.0f});
              ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                    running_ ? ImVec4{0.13f, 0.28f, 0.43f, 1.0f} : ImVec4{0.10f, 0.30f, 0.21f, 1.0f});
              if (ImGui::Button(running_ ? "Pause" : "Resume", button_size))
                running_ = !running_;
              ImGui::PopStyleColor(3);
              const float label_width =
                  std::max(ImGui::CalcTextSize("Galaxy Number").x,
                           ImGui::GetFrameHeight() + style.ItemInnerSpacing.x + ImGui::CalcTextSize("VSync").x);
              const float field_width = row_width - label_width - style.ItemSpacing.x;
              ImGui::SetNextItemWidth(field_width);
              ImGui::SliderFloat("##DeltaTime", &delta_t_, 0.001f, 0.1f, "%.3f", ImGuiSliderFlags_Logarithmic);
              ImGui::SameLine();
              ImGui::TextUnformatted("Delta Time");
              ImGui::SetNextItemWidth(field_width);
              ImGui::SliderInt("##GalaxyNumber", &galaxy_count_, 1, 20);
              ImGui::SameLine();
              ImGui::TextUnformatted("Galaxy Number");
              ImGui::SeparatorText("ColorSpace");
              ImGui::SetNextItemWidth(field_width);
              if (ImGui::BeginCombo("##ColorSpace", ColorSpaceName(window_->GetColorSpace()).c_str())) {
                for (auto space : spaces_) {
                  if (ImGui::Selectable(ColorSpaceName(space).c_str(), space == window_->GetColorSpace())) {
                    requested_space_ = space;
                    output_change_requested_ = true;
                  }
                }
                ImGui::EndCombo();
              }
              ImGui::SameLine();
              bool vsync = requested_ == PRESENT_MODE_VSYNC;
              if (ImGui::Checkbox("VSync", &vsync)) {
                requested_ = vsync ? PRESENT_MODE_VSYNC : PRESENT_MODE_CONTINUOUS;
                present_mode_changed_ = true;
              }
              if (!status_.empty())
                ImGui::TextUnformatted(status_.c_str());
              ImGui::SeparatorText("Speed Distribution");
              if (speeds_ready_) {
                ImGui::PlotLines("##Speeds", normalized_speeds_.data(), int(normalized_speeds_.size()), 0, nullptr, 0,
                                 1, {ImGui::GetContentRegionAvail().x, 80});
                ImGui::Text("Speed: 0 - %.3f", maximum_speed_);
              } else
                ImGui::TextUnformatted("Waiting for GPU readback...");
              ImGui::SeparatorText("Instructions");
              ImGui::TextWrapped("Galaxy count applies on Reset. Drag outside UI to rotate.");
              ImGui::TextWrapped("I: show/hide GUI; P: VSync; Space: pause/resume; R: reset; Escape: close");
            }
            ImGui::End();
            ImGui::PopStyleVar(2);
          }
          capture_keyboard_ = show_gui_ && ImGui::GetIO().WantCaptureKeyboard;
          if (!show_gui_ || !ImGui::GetIO().WantCaptureMouse) {
            rotation_ =
                glm::rotate(glm::mat4{1}, glm::radians(float(rotation_delta_.x)), glm::vec3{0, 1, 0}) * rotation_;
            rotation_ =
                glm::rotate(glm::mat4{1}, glm::radians(float(rotation_delta_.y)), glm::vec3{1, 0, 0}) * rotation_;
          }
          rotation_delta_ = {};
        }))
      return 1;
    if (output_change_requested_) {
      const auto next = requested_space_;
      output_change_requested_ = false;
      if (next == window_->GetColorSpace())
        status_.clear();
      else if (!window_->SetColorSpace(next))
        status_ = "HDR mode unavailable; keeping the current output.";
      else {
        status_.clear();
        fps_.Reset();
        std::cout << "nbody: output " << ColorSpaceName(next) << '\n';
      }
    }
    if (window_->IsSwapChainInvalid()) {
      if (!window_->RebuildSwapChain())
        return 1;
    }
    if (present_mode_changed_) {
      present_mode_changed_ = false;
      if (!window_->SetPresentMode(requested_))
        return 1;
    }
  }

  return 0;
}

int NBodyModule::OnRender() {
  const size_t slot = (static_cast<int64_t>(GetFrameIndex()) + 60) % targets_.size();
  if (headless_) {
    if (GetFrameIndex() == 0) {
      for (auto &token : pending_) {
        if (token) {
          if (!token->WaitHost())
            return 1;
          token = {};
        }
      }
      benchmark_start_ = std::chrono::steady_clock::now();
    }
    if (pending_[slot]) {
      if (!pending_[slot]->WaitHost())
        return 1;
      pending_[slot] = {};
    }
  }
  ObjPtr<IImage> image;
  if (headless_)
    image = targets_[slot];
  else
    image = window_->GetNextImage();
  if (!image) {
    if (headless_)
      return 1;
    fps_.Reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return 0;
  }
  auto extent = image->GetDesc();
  if (!radiance_ || radiance_->GetDesc().width != extent.width || radiance_->GetDesc().height != extent.height) {
    radiance_ = device_->CreateImage({extent.width, extent.height, IMAGE_FORMAT_R32G32B32A32_SFLOAT});
    if (!radiance_ || !output_group_->BindResource(1, radiance_.Get()))
      return 1;
  }
  if (headless_)
    frame_image_ = image;
  else if (!frame_image_ || frame_image_->GetDesc().width != extent.width ||
           frame_image_->GetDesc().height != extent.height) {
    frame_image_ = device_->CreateImage({extent.width, extent.height, IMAGE_FORMAT_R32G32B32A32_SFLOAT});
    if (!frame_image_)
      return 1;
  }
  if (!output_program_) {
    const auto &code = output_source_;
    ObjPtr<IShader> vs = device_->CreateShaderFromSource(code, "VSMain", SHADER_STAGE_VERTEX);
    ObjPtr<IShader> ps = device_->CreateShaderFromSource(code, "PSMain", SHADER_STAGE_FRAGMENT);
    if (!vs || !ps)
      return 1;
    ProgramDesc output_desc;
    output_desc.vertex_shader = vs.Get();
    output_desc.fragment_shader = ps.Get();
    output_desc.color_formats = {IMAGE_FORMAT_R32G32B32A32_SFLOAT};
    output_desc.pipeline_layout.groups = {output_layout_};
    output_program_ = device_->CreateProgram(output_desc);
    if (!output_program_)
      return 1;
  }
  ObjPtr<ICommandContext> context = device_->CreateCommandContext();
  if (!context)
    return 1;
  const auto view = glm::lookAt(glm::vec3(10, 20, 30), glm::vec3(0), glm::vec3(0, 1, 0)) * rotation_;
  Camera camera{glm::perspective(glm::radians(60.0f), float(extent.width) / extent.height, 0.1f, 100.0f) * view,
                glm::inverse(view), 0.07f};
  if (!context->CmdUploadBuffer(camera_uniform_.Get(), 0, &camera, sizeof(camera)))
    return 1;
  if (reset_requested_) {
    if (!ResetParticles(context.Get()))
      return 1;
    reset_requested_ = false;
    std::cout << "nbody: reset " << galaxy_count_ << " galaxies\n";
  }
  if (running_) {
    const Simulation settings{particle_count, delta_t_, 100.0f / particle_count};
    if (!context->CmdUploadBuffer(simulation_uniform_.Get(), 0, &settings, sizeof(settings)) ||
        !context->CmdBindComputeProgram(compute_.Get()) || !context->CmdBindGroup(0, simulation_group_.Get()) ||
        !context->CmdDispatch(particle_count / 128) ||
        !context->CmdCopyBuffer(next_positions_.Get(), 0, positions_.Get(), 0, bytes_))
      return 1;
  }
  if (!context->CmdBindProgram(program_.Get()))
    return 1;
  if (!context->CmdBindGroup(0, camera_group_.Get()))
    return 1;
  if (!context->CmdBeginRendering(
          {{{radiance_.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}))
    return 1;
  if (!context->CmdBindVertexBuffer(0, positions_.Get()))
    return 1;
  if (!context->CmdSetViewport({0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1}))
    return 1;
  if (!context->CmdSetScissor({0, 0, extent.width, extent.height}))
    return 1;
  if (!context->CmdDraw(6, particle_count))
    return 1;
  if (!context->CmdEndRendering())
    return 1;
  const float display[4] = {headless_ ? 1.0f : window_->GetHDRHeadroom(), 0, 0, 0};
  if (!context->CmdUploadBuffer(output_uniform_.Get(), 0, display, sizeof(display)) ||
      !context->CmdBindProgram(output_program_.Get()) || !context->CmdBindGroup(0, output_group_.Get()) ||
      !context->CmdBeginRendering({{{frame_image_.Get(), ATTACHMENT_LOAD_OP_DISCARD, ATTACHMENT_STORE_OP_STORE}}}) ||
      !context->CmdSetViewport({0, 0, float(extent.width), float(extent.height), 0, 1}) ||
      !context->CmdSetScissor({0, 0, extent.width, extent.height}) || !context->CmdDraw(3) ||
      !context->CmdEndRendering())
    return 1;
  if (!headless_) {
    if (!context->CmdDrawImGui(window_.Get(), frame_image_.Get()) ||
        !context->CmdPresentImage(window_.Get(), frame_image_.Get(), COLOR_SPACE_BT709_LINEAR))
      return 1;
  }
  if (!velocity_download_) {
    velocity_download_ = context->CmdDownloadBuffer(velocities_.Get(), 0, particle_count * sizeof(glm::vec3));
    if (!velocity_download_)
      return 1;
  }
  if (!capture_directory_.empty()) {
    last_frame_ = context->CmdDownloadImage(image.Get());
    if (!last_frame_)
      return 1;
    last_frame_desc_ = image->GetDesc();
    last_frame_space_ = window_ ? window_->GetColorSpace() : COLOR_SPACE_BT709_LINEAR;
    last_frame_white_scale_ = window_ ? window_->GetHDRReferenceWhiteScale() : 1.0f;
  }
  ObjPtr<ISubmissionToken> token = queue_->Submit(context.Get());

  if (!token)
    return 1;
  if (headless_)
    pending_[slot] = std::move(token);
  CompleteFrame();
  fps_.TickFrame();

  return 0;
}

int NBodyModule::OnTerminate() {
  int result = 0;
  if (initialized_) {
    result = [&]() -> int {
      if (headless_) {
        for (const auto &token : pending_) {
          if (token) {
            if (!token->WaitHost())
              return 1;
          }
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - benchmark_start_).count();
        std::cout << fmt::format("nbody: headless 1280x720, {} frames, {:.6f} s, {:.2f} FPS (3 in flight, 60 warmup)\n",
                                 GetFrameIndex(), seconds, GetFrameIndex() / seconds);
        std::cout << fmt::format(
            "nbody: {} particles, {} FLOP/frame, estimated {:.3f} TFLOP/s (20 FLOP/interaction, whole-frame time)\n",
            particle_count, flops_per_frame, flops_per_frame * (GetFrameIndex() / seconds) * 1e-12);
      }
      if (velocity_download_ && !velocity_download_->WaitHost())
        return 1;
      if (!UpdateSpeeds())
        return 1;
      if (speeds_ready_) {
        std::cout << "nbody: downloaded " << particle_count << " particle velocities, max speed " << maximum_speed_
                  << ", 100 histogram bins\n";
        if (!capture_directory_.empty()) {
          std::error_code error;
          const auto path = std::filesystem::u8path(capture_directory_);
          std::filesystem::create_directories(path, error);
          if (error)
            return 1;
          std::ofstream file(path / "nbody_speeds.csv");
          file << "speed_min,speed_max,count,normalized\n";
          for (size_t i = 0; i < speed_counts_.size(); ++i)
            file << maximum_speed_ * i / speed_counts_.size() << ',' << maximum_speed_ * (i + 1) / speed_counts_.size()
                 << ',' << speed_counts_[i] << ',' << normalized_speeds_[i] << '\n';
          file.close();
          if (!file)
            return 1;
        }
      }
      if (!capture_directory_.empty()) {
        if (!last_frame_ || !last_frame_->WaitHost())
          return 1;
        std::vector<uint8_t> pixels(last_frame_->GetSize());
        if (!last_frame_->CopyTo(pixels.data(), pixels.size()) ||
            !SaveImage(capture_directory_, "nbody", pixels, last_frame_desc_, last_frame_space_,
                       last_frame_white_scale_))
          return 1;
      }
      return 0;
    }();
  }
  benchmark_start_ = {};
  fps_ = {};
  speeds_ready_ = {};
  maximum_speed_ = {};
  normalized_speeds_ = {};
  speed_counts_ = {};
  downloaded_velocities_ = {};
  velocity_download_ = {};
  random_ = {};
  camera_group_ = {};
  simulation_group_ = {};
  camera_uniform_ = {};
  simulation_uniform_ = {};
  next_positions_ = {};
  velocities_ = {};
  positions_ = {};
  bytes_ = {};
  compute_ = {};
  output_program_ = {};
  output_group_ = {};
  output_uniform_ = {};
  output_layout_ = {};
  output_source_ = {};
  frame_image_ = {};
  radiance_ = {};
  program_ = {};
  pending_ = {};
  targets_ = {};
  queue_ = {};
  window_ = {};
  spaces_ = {};
  requested_space_ = {};
  show_gui_ = {};
  output_change_requested_ = {};
  capture_keyboard_ = {};
  status_ = {};
  dragging_ = {};
  cursor_initialized_ = {};
  rotation_delta_ = {};
  last_cursor_ = {};
  rotation_ = {};
  galaxy_count_ = {};
  delta_t_ = {};
  reset_requested_ = {};
  running_ = {};
  present_mode_changed_ = {};
  requested_ = {};
  last_frame_white_scale_ = {};
  last_frame_space_ = {};
  last_frame_desc_ = {};
  last_frame_ = {};
  device_ = {};
  api_ = {};
  capture_directory_ = {};
  headless_ = {};
  initialized_ = {};
  return result;
}

bool NBodyModule::ResetParticles(grassland::rhi::ICommandContext *upload) {
  auto unit = [&]() { return std::uniform_real_distribution<float>()(random_); };
  auto sphere = [&]() {
    const float z = unit() * 2 - 1;
    const float radius = std::sqrt(1 - z * z);
    const float angle = unit() * 6.28318530718f;
    return glm::vec3(radius * std::sin(angle), radius * std::cos(angle), z) * std::cbrt(unit());
  };
  std::vector<glm::vec3> origins(galaxy_count_), speeds(galaxy_count_);
  glm::vec3 mean_position(0), mean_velocity(0);
  for (size_t i = 0; i < origins.size(); ++i) {
    origins[i] = sphere() * 20.0f;
    speeds[i] = sphere();
    mean_position += origins[i];
    mean_velocity += speeds[i];
  }
  mean_position /= float(origins.size());
  mean_velocity /= float(origins.size());
  std::vector<glm::vec3> initial_positions(particle_count), initial_velocities(particle_count);
  for (uint32_t i = 0; i < particle_count; ++i) {
    auto galaxy = std::uniform_int_distribution<int>(0, galaxy_count_ - 1)(random_);
    initial_positions[i] = sphere() * (2.0f * std::cbrt(10.0f / galaxy_count_)) + origins[galaxy] - mean_position;
    initial_velocities[i] = sphere() * 2.0f + speeds[galaxy] - mean_velocity;
  }
  return upload->CmdUploadBuffer(positions_.Get(), 0, initial_positions.data(), bytes_) &&
         upload->CmdUploadBuffer(velocities_.Get(), 0, initial_velocities.data(), bytes_);
}

bool NBodyModule::UpdateSpeeds() {
  if (!velocity_download_)
    return true;
  const auto status = velocity_download_->GetStatus();
  if (status == SUBMISSION_STATUS_PENDING)
    return true;
  if (status != SUBMISSION_STATUS_COMPLETED ||
      !velocity_download_->CopyTo(downloaded_velocities_.data(), downloaded_velocities_.size() * sizeof(glm::vec3)))
    return false;
  velocity_download_ = {};
  maximum_speed_ = 0;
  for (const auto &velocity : downloaded_velocities_) {
    const float speed = glm::length(velocity);
    if (!std::isfinite(speed)) {
      std::cerr << "nbody: non-finite particle speed\n";
      return false;
    }
    maximum_speed_ = std::max(maximum_speed_, speed);
  }
  speed_counts_.fill(0);
  for (const auto &velocity : downloaded_velocities_) {
    const size_t bin =
        maximum_speed_ > 0
            ? std::min(size_t(glm::length(velocity) / maximum_speed_ * speed_counts_.size()), speed_counts_.size() - 1)
            : 0;
    ++speed_counts_[bin];
  }
  const auto peak = *std::max_element(speed_counts_.begin(), speed_counts_.end());
  for (size_t i = 0; i < speed_counts_.size(); ++i)
    normalized_speeds_[i] = peak ? float(speed_counts_[i]) / peak : 0;
  speeds_ready_ = true;
  return true;
}
