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

struct NBodyModule::State {
  bool initialized{};
  bool headless{};
  std::string capture_directory{};
  grassland::rhi::BackendAPI api{};
  grassland::rhi::IDevice *device{};
  ObjPtr<IDownloadToken> last_frame{};
  ImageDesc last_frame_desc{};
  ColorSpace last_frame_space{};
  float last_frame_white_scale{};
  PresentMode requested{};
  bool present_mode_changed{};
  bool running{};
  bool reset_requested{};
  float delta_t{};
  int galaxy_count{};
  glm::mat4 rotation{};
  glm::dvec2 last_cursor{};
  glm::dvec2 rotation_delta{};
  bool cursor_initialized{};
  bool dragging{};
  std::string status{};
  bool capture_keyboard{};
  bool output_change_requested{};
  bool show_gui{};
  ColorSpace requested_space{};
  std::vector<ColorSpace> spaces{};
  ObjPtr<IWindow> window{};
  ObjPtr<IQueue> queue{};
  std::array<ObjPtr<IImage>, 3> targets{};
  std::array<ObjPtr<ISubmissionToken>, 3> pending{};
  ObjPtr<IProgram> program{};
  ObjPtr<IImage> radiance{};
  ObjPtr<IImage> frame_image{};
  std::string output_source{};
  BindingLayoutDesc output_layout{};
  ObjPtr<IBuffer> output_uniform{};
  ObjPtr<IBindingGroup> output_group{};
  ObjPtr<IProgram> output_program{};
  ObjPtr<IComputeProgram> compute{};
  uint64_t bytes{};
  ObjPtr<IBuffer> positions{};
  ObjPtr<IBuffer> velocities{};
  ObjPtr<IBuffer> next_positions{};
  ObjPtr<IBuffer> simulation_uniform{};
  ObjPtr<IBuffer> camera_uniform{};
  ObjPtr<IBindingGroup> simulation_group{};
  ObjPtr<IBindingGroup> camera_group{};
  std::mt19937 random{};
  ObjPtr<IDownloadToken> velocity_download{};
  std::vector<glm::vec3> downloaded_velocities{};
  std::array<uint32_t, 100> speed_counts{};
  std::array<float, 100> normalized_speeds{};
  float maximum_speed{};
  bool speeds_ready{};
  grassland::FPSCounter fps{};
  std::chrono::steady_clock::time_point benchmark_start{};
};

NBodyModule::NBodyModule(grassland::rhi::ObjPtr<grassland::rhi::ICore> core,
                         grassland::rhi::ObjPtr<grassland::rhi::IDevice> device)
    : ModuleBase(std::move(core), std::move(device)) {
}

NBodyModule::~NBodyModule() = default;

int NBodyModule::OnInit(const ModuleOptions &options) {
  state_ = std::make_unique<State>();

  state_->headless = options.headless;
  state_->capture_directory = options.capture_directory;
  state_->api = GetBackendAPI();
  using namespace grassland::rhi;
  state_->device = GetDevice();

  state_->last_frame_desc = ImageDesc{};
  state_->last_frame_space = COLOR_SPACE_SRGB_NONLINEAR;
  state_->last_frame_white_scale = 1.0f;
  state_->requested = PRESENT_MODE_VSYNC;
  state_->present_mode_changed = false;
  state_->running = true;
  state_->reset_requested = true;
  state_->delta_t = 0.03f;
  state_->galaxy_count = 10;
  state_->rotation = glm::mat4{1.0f};
  state_->last_cursor = glm::dvec2{};
  state_->rotation_delta = glm::dvec2{};
  state_->cursor_initialized = false;
  state_->dragging = false;

  state_->capture_keyboard = false;
  state_->output_change_requested = false;
  state_->show_gui = true;
  state_->requested_space = COLOR_SPACE_SRGB_NONLINEAR;
  state_->spaces = std::vector<ColorSpace>{COLOR_SPACE_SRGB_NONLINEAR};

  if (!state_->headless) {
    state_->window = state_->device->CreateGLFWWindow({1280, 720, "RHI nbody"});
    if (!state_->window)
      return 1;
    if (!state_->window->EnableImGui(true, LONGMARCH_ASSETS_DIR "fonts/simhei.ttf", 20.0f))
      return 1;
    for (const auto &option : state_->window->GetSupportedPresentationOptions()) {
      if (std::find(state_->spaces.begin(), state_->spaces.end(), option.color_space) == state_->spaces.end())
        state_->spaces.push_back(option.color_space);
    }
  }
  state_->queue = state_->device->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!state_->queue)
    return 1;

  if (state_->headless) {
    for (auto &target : state_->targets) {
      target = state_->device->CreateImage({1280, 720, IMAGE_FORMAT_R32G32B32A32_SFLOAT});
      if (!target)
        return 1;
    }
  }
  const auto &embedded = shader_list.at("modules/nbody/shaders/particle.slang");
  std::string source(embedded.first, embedded.second);
  ObjPtr<IShader> vertex_shader = state_->device->CreateShaderFromSource(source, "VSMain", SHADER_STAGE_VERTEX);
  if (!vertex_shader)
    return 1;
  ObjPtr<IShader> fragment_shader = state_->device->CreateShaderFromSource(source, "PSMain", SHADER_STAGE_FRAGMENT);
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
  state_->program = state_->device->CreateProgram(desc);
  if (!state_->program)
    return 1;

  const auto &output_embedded = shader_list.at("modules/nbody/shaders/present.slang");
  state_->output_source = std::string(output_embedded.first, output_embedded.second);
  state_->output_layout = BindingLayoutDesc{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT},
                                             {1, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT}}};
  state_->output_uniform = state_->device->CreateBuffer({256});
  state_->output_group = state_->device->CreateBindingGroup(state_->output_layout);
  if (!state_->output_uniform || !state_->output_group ||
      !state_->output_group->BindResource(0, state_->output_uniform.Get()))
    return 1;

  const auto &compute_source = shader_list.at("modules/nbody/shaders/nbody.slang");
  ObjPtr<IShader> compute_shader = state_->device->CreateShaderFromSource(
      std::string(compute_source.first, compute_source.second), "CSMain", SHADER_STAGE_COMPUTE);
  if (!compute_shader)
    return 1;
  ComputeProgramDesc compute_desc;
  compute_desc.compute_shader = compute_shader.Get();
  compute_desc.pipeline_layout.groups = {{{{0, BINDING_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                                           {1, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                                           {2, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                                           {3, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_COMPUTE}}}};
  state_->compute = state_->device->CreateComputeProgram(compute_desc);
  if (!state_->compute)
    return 1;
  state_->bytes = particle_count * sizeof(glm::vec3);
  state_->positions = state_->device->CreateBuffer({state_->bytes});
  if (!state_->positions)
    return 1;
  state_->velocities = state_->device->CreateBuffer({state_->bytes});
  if (!state_->velocities)
    return 1;
  state_->next_positions = state_->device->CreateBuffer({state_->bytes});
  if (!state_->next_positions)
    return 1;
  state_->simulation_uniform = state_->device->CreateBuffer({256});
  if (!state_->simulation_uniform)
    return 1;
  state_->camera_uniform = state_->device->CreateBuffer({256});
  if (!state_->camera_uniform)
    return 1;
  state_->simulation_group = state_->device->CreateBindingGroup(compute_desc.pipeline_layout.groups[0]);
  if (!state_->simulation_group)
    return 1;
  if (!state_->simulation_group->BindResource(0, state_->positions.Get()))
    return 1;
  if (!state_->simulation_group->BindResource(1, state_->velocities.Get()))
    return 1;
  if (!state_->simulation_group->BindResource(2, state_->next_positions.Get()))
    return 1;
  if (!state_->simulation_group->BindResource(3, state_->simulation_uniform.Get()))
    return 1;
  state_->camera_group = state_->device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  if (!state_->camera_group)
    return 1;
  if (!state_->camera_group->BindResource(0, state_->camera_uniform.Get()))
    return 1;
  state_->random = std::mt19937(42);

  if (!state_->headless) {
    state_->window->KeyEvent().RegisterCallback([this](int key, int, int action, int) {
      if (action != GLFW_PRESS)
        return;
      if (key == GLFW_KEY_I) {
        state_->show_gui = !state_->show_gui;
        state_->capture_keyboard = false;
        return;
      }
      if (state_->capture_keyboard && key != GLFW_KEY_ESCAPE)
        return;
      if (key == GLFW_KEY_SPACE)
        state_->running = !state_->running;
      if (key == GLFW_KEY_R)
        state_->reset_requested = true;
      if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(state_->window->GLFWWindow(), GLFW_TRUE);
      if (key == GLFW_KEY_P) {
        state_->requested = state_->requested == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
        state_->present_mode_changed = true;
      }
    });
  }
  if (state_->window) {
    state_->window->MouseButtonEvent().RegisterCallback([this](int button, int action, int, double, double) {
      if (button == GLFW_MOUSE_BUTTON_LEFT)
        state_->dragging = action != GLFW_RELEASE;
    });
    state_->window->MouseMoveEvent().RegisterCallback([this](double x, double y) {
      const glm::dvec2 cursor{x, y};
      if (state_->cursor_initialized && state_->dragging)
        state_->rotation_delta += cursor - state_->last_cursor;
      state_->last_cursor = cursor;
      state_->cursor_initialized = true;
    });
  }

  state_->downloaded_velocities = std::vector<glm::vec3>(particle_count);
  state_->speed_counts = std::array<uint32_t, 100>{};
  state_->normalized_speeds = std::array<float, 100>{};
  state_->maximum_speed = 0;
  state_->speeds_ready = false;

  // Warm up allocations and pipelines before measuring completed GPU work.
  state_->benchmark_start = std::chrono::steady_clock::now();
  SetWarmupFrames(state_->headless ? 60 : 0);

  state_->initialized = true;
  return 0;
}

int NBodyModule::OnUpdate() {
  if (!UpdateSpeeds())
    return 1;
  if (!state_->headless) {
    if (state_->window->ShouldClose()) {
      RequestClose();
      return 0;
    }
    if (!state_->window->UpdateImGui([&] {
          if (state_->show_gui) {
            ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Once);
            ImGui::SetNextWindowBgAlpha(0.9f);
            ImGui::SetNextWindowSize({ImGui::GetFontSize() * 30.0f, 0}, ImGuiCond_Always);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {8.0f, 6.0f});
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
            if (ImGui::Begin(
                    "NBody", nullptr,
                    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize)) {
              ImGui::SeparatorText("Statistics");
              ImGui::Text("Backend: %s", BackendAPIName(state_->api).c_str());
              ImGui::Text("Particles: %u", particle_count);
              const double rate = state_->fps.GetFPS();
              ImGui::Text("FPS: %.1f", rate);
              ImGui::Text("Frame Duration: %.3f ms", rate > 0 ? 1000.0 / rate : 0.0);
              ImGui::Text("Estimated: %.2f TFLOP/s", state_->running ? flops_per_frame * rate * 1e-12 : 0.0);
              ImGui::SeparatorText("Control");
              const auto &style = ImGui::GetStyle();
              const float row_width = ImGui::GetContentRegionAvail().x;
              const ImVec2 button_size{(row_width - style.ItemSpacing.x) * 0.5f, ImGui::GetFrameHeight() + 6.0f};
              ImGui::PushStyleColor(ImGuiCol_Button, {0.20f, 0.25f, 0.32f, 1.0f});
              ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.28f, 0.35f, 0.44f, 1.0f});
              ImGui::PushStyleColor(ImGuiCol_ButtonActive, {0.33f, 0.41f, 0.52f, 1.0f});
              if (ImGui::Button("Reset", button_size))
                state_->reset_requested = true;
              ImGui::PopStyleColor(3);
              ImGui::SameLine();
              ImGui::PushStyleColor(ImGuiCol_Button, state_->running ? ImVec4{0.16f, 0.34f, 0.52f, 1.0f}
                                                                     : ImVec4{0.14f, 0.38f, 0.28f, 1.0f});
              ImGui::PushStyleColor(ImGuiCol_ButtonHovered, state_->running ? ImVec4{0.20f, 0.43f, 0.65f, 1.0f}
                                                                            : ImVec4{0.18f, 0.49f, 0.36f, 1.0f});
              ImGui::PushStyleColor(ImGuiCol_ButtonActive, state_->running ? ImVec4{0.13f, 0.28f, 0.43f, 1.0f}
                                                                           : ImVec4{0.10f, 0.30f, 0.21f, 1.0f});
              if (ImGui::Button(state_->running ? "Pause" : "Resume", button_size))
                state_->running = !state_->running;
              ImGui::PopStyleColor(3);
              const float label_width =
                  std::max(ImGui::CalcTextSize("Galaxy Number").x,
                           ImGui::GetFrameHeight() + style.ItemInnerSpacing.x + ImGui::CalcTextSize("VSync").x);
              const float field_width = row_width - label_width - style.ItemSpacing.x;
              ImGui::SetNextItemWidth(field_width);
              ImGui::SliderFloat("##DeltaTime", &state_->delta_t, 0.001f, 0.1f, "%.3f", ImGuiSliderFlags_Logarithmic);
              ImGui::SameLine();
              ImGui::TextUnformatted("Delta Time");
              ImGui::SetNextItemWidth(field_width);
              ImGui::SliderInt("##GalaxyNumber", &state_->galaxy_count, 1, 20);
              ImGui::SameLine();
              ImGui::TextUnformatted("Galaxy Number");
              ImGui::SeparatorText("ColorSpace");
              ImGui::SetNextItemWidth(field_width);
              if (ImGui::BeginCombo("##ColorSpace", ColorSpaceName(state_->window->GetColorSpace()).c_str())) {
                for (auto space : state_->spaces) {
                  if (ImGui::Selectable(ColorSpaceName(space).c_str(), space == state_->window->GetColorSpace())) {
                    state_->requested_space = space;
                    state_->output_change_requested = true;
                  }
                }
                ImGui::EndCombo();
              }
              ImGui::SameLine();
              bool vsync = state_->requested == PRESENT_MODE_VSYNC;
              if (ImGui::Checkbox("VSync", &vsync)) {
                state_->requested = vsync ? PRESENT_MODE_VSYNC : PRESENT_MODE_CONTINUOUS;
                state_->present_mode_changed = true;
              }
              if (!state_->status.empty())
                ImGui::TextUnformatted(state_->status.c_str());
              ImGui::SeparatorText("Speed Distribution");
              if (state_->speeds_ready) {
                ImGui::PlotLines("##Speeds", state_->normalized_speeds.data(), int(state_->normalized_speeds.size()), 0,
                                 nullptr, 0, 1, {ImGui::GetContentRegionAvail().x, 80});
                ImGui::Text("Speed: 0 - %.3f", state_->maximum_speed);
              } else
                ImGui::TextUnformatted("Waiting for GPU readback...");
              ImGui::SeparatorText("Instructions");
              ImGui::TextWrapped("Galaxy count applies on Reset. Drag outside UI to rotate.");
              ImGui::TextWrapped("I: show/hide GUI; P: VSync; Space: pause/resume; R: reset; Escape: close");
            }
            ImGui::End();
            ImGui::PopStyleVar(2);
          }
          state_->capture_keyboard = state_->show_gui && ImGui::GetIO().WantCaptureKeyboard;
          if (!state_->show_gui || !ImGui::GetIO().WantCaptureMouse) {
            state_->rotation =
                glm::rotate(glm::mat4{1}, glm::radians(float(state_->rotation_delta.x)), glm::vec3{0, 1, 0}) *
                state_->rotation;
            state_->rotation =
                glm::rotate(glm::mat4{1}, glm::radians(float(state_->rotation_delta.y)), glm::vec3{1, 0, 0}) *
                state_->rotation;
          }
          state_->rotation_delta = {};
        }))
      return 1;
    if (state_->output_change_requested) {
      const auto next = state_->requested_space;
      state_->output_change_requested = false;
      if (next == state_->window->GetColorSpace())
        state_->status.clear();
      else if (!state_->window->SetColorSpace(next))
        state_->status = "HDR mode unavailable; keeping the current output.";
      else {
        state_->status.clear();
        state_->fps.Reset();
        std::cout << "nbody: output " << ColorSpaceName(next) << '\n';
      }
    }
    if (state_->window->IsSwapChainInvalid()) {
      if (!state_->window->RebuildSwapChain())
        return 1;
    }
    if (state_->present_mode_changed) {
      state_->present_mode_changed = false;
      if (!state_->window->SetPresentMode(state_->requested))
        return 1;
    }
  }

  return 0;
}

int NBodyModule::OnRender() {
  const size_t slot = (static_cast<int64_t>(GetFrameIndex()) + 60) % state_->targets.size();
  if (state_->headless) {
    if (GetFrameIndex() == 0) {
      for (auto &token : state_->pending) {
        if (token) {
          if (!token->WaitHost())
            return 1;
          token = {};
        }
      }
      state_->benchmark_start = std::chrono::steady_clock::now();
    }
    if (state_->pending[slot]) {
      if (!state_->pending[slot]->WaitHost())
        return 1;
      state_->pending[slot] = {};
    }
  }
  ObjPtr<IImage> image;
  if (state_->headless)
    image = state_->targets[slot];
  else
    image = state_->window->GetNextImage();
  if (!image) {
    if (state_->headless)
      return 1;
    state_->fps.Reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return 0;
  }
  auto extent = image->GetDesc();
  if (!state_->radiance || state_->radiance->GetDesc().width != extent.width ||
      state_->radiance->GetDesc().height != extent.height) {
    state_->radiance = state_->device->CreateImage({extent.width, extent.height, IMAGE_FORMAT_R32G32B32A32_SFLOAT});
    if (!state_->radiance || !state_->output_group->BindResource(1, state_->radiance.Get()))
      return 1;
  }
  if (state_->headless)
    state_->frame_image = image;
  else if (!state_->frame_image || state_->frame_image->GetDesc().width != extent.width ||
           state_->frame_image->GetDesc().height != extent.height) {
    state_->frame_image = state_->device->CreateImage({extent.width, extent.height, IMAGE_FORMAT_R32G32B32A32_SFLOAT});
    if (!state_->frame_image)
      return 1;
  }
  if (!state_->output_program) {
    const auto &code = state_->output_source;
    ObjPtr<IShader> vs = state_->device->CreateShaderFromSource(code, "VSMain", SHADER_STAGE_VERTEX);
    ObjPtr<IShader> ps = state_->device->CreateShaderFromSource(code, "PSMain", SHADER_STAGE_FRAGMENT);
    if (!vs || !ps)
      return 1;
    ProgramDesc output_desc;
    output_desc.vertex_shader = vs.Get();
    output_desc.fragment_shader = ps.Get();
    output_desc.color_formats = {IMAGE_FORMAT_R32G32B32A32_SFLOAT};
    output_desc.pipeline_layout.groups = {state_->output_layout};
    state_->output_program = state_->device->CreateProgram(output_desc);
    if (!state_->output_program)
      return 1;
  }
  ObjPtr<ICommandContext> context = state_->device->CreateCommandContext();
  if (!context)
    return 1;
  const auto view = glm::lookAt(glm::vec3(10, 20, 30), glm::vec3(0), glm::vec3(0, 1, 0)) * state_->rotation;
  Camera camera{glm::perspective(glm::radians(60.0f), float(extent.width) / extent.height, 0.1f, 100.0f) * view,
                glm::inverse(view), 0.07f};
  if (!context->CmdUploadBuffer(state_->camera_uniform.Get(), 0, &camera, sizeof(camera)))
    return 1;
  if (state_->reset_requested) {
    if (!ResetParticles(context.Get()))
      return 1;
    state_->reset_requested = false;
    std::cout << "nbody: reset " << state_->galaxy_count << " galaxies\n";
  }
  if (state_->running) {
    const Simulation settings{particle_count, state_->delta_t, 100.0f / particle_count};
    if (!context->CmdUploadBuffer(state_->simulation_uniform.Get(), 0, &settings, sizeof(settings)) ||
        !context->CmdBindComputeProgram(state_->compute.Get()) ||
        !context->CmdBindGroup(0, state_->simulation_group.Get()) || !context->CmdDispatch(particle_count / 128) ||
        !context->CmdCopyBuffer(state_->next_positions.Get(), 0, state_->positions.Get(), 0, state_->bytes))
      return 1;
  }
  if (!context->CmdBindProgram(state_->program.Get()))
    return 1;
  if (!context->CmdBindGroup(0, state_->camera_group.Get()))
    return 1;
  if (!context->CmdBeginRendering(
          {{{state_->radiance.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}))
    return 1;
  if (!context->CmdBindVertexBuffer(0, state_->positions.Get()))
    return 1;
  if (!context->CmdSetViewport({0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1}))
    return 1;
  if (!context->CmdSetScissor({0, 0, extent.width, extent.height}))
    return 1;
  if (!context->CmdDraw(6, particle_count))
    return 1;
  if (!context->CmdEndRendering())
    return 1;
  const float display[4] = {state_->headless ? 1.0f : state_->window->GetHDRHeadroom(), 0, 0, 0};
  if (!context->CmdUploadBuffer(state_->output_uniform.Get(), 0, display, sizeof(display)) ||
      !context->CmdBindProgram(state_->output_program.Get()) || !context->CmdBindGroup(0, state_->output_group.Get()) ||
      !context->CmdBeginRendering(
          {{{state_->frame_image.Get(), ATTACHMENT_LOAD_OP_DISCARD, ATTACHMENT_STORE_OP_STORE}}}) ||
      !context->CmdSetViewport({0, 0, float(extent.width), float(extent.height), 0, 1}) ||
      !context->CmdSetScissor({0, 0, extent.width, extent.height}) || !context->CmdDraw(3) ||
      !context->CmdEndRendering())
    return 1;
  if (!state_->headless) {
    if (!context->CmdDrawImGui(state_->window.Get(), state_->frame_image.Get()) ||
        !context->CmdPresentImage(state_->window.Get(), state_->frame_image.Get(), COLOR_SPACE_BT709_LINEAR))
      return 1;
  }
  if (!state_->velocity_download) {
    state_->velocity_download =
        context->CmdDownloadBuffer(state_->velocities.Get(), 0, particle_count * sizeof(glm::vec3));
    if (!state_->velocity_download)
      return 1;
  }
  if (!state_->capture_directory.empty()) {
    state_->last_frame = context->CmdDownloadImage(image.Get());
    if (!state_->last_frame)
      return 1;
    state_->last_frame_desc = image->GetDesc();
    state_->last_frame_space = state_->window ? state_->window->GetColorSpace() : COLOR_SPACE_BT709_LINEAR;
    state_->last_frame_white_scale = state_->window ? state_->window->GetHDRReferenceWhiteScale() : 1.0f;
  }
  ObjPtr<ISubmissionToken> token = state_->queue->Submit(context.Get());

  if (!token)
    return 1;
  if (state_->headless)
    state_->pending[slot] = std::move(token);
  CompleteFrame();
  state_->fps.TickFrame();

  return 0;
}

int NBodyModule::OnTerminate() {
  int result = 0;
  if (state_ && state_->initialized) {
    result = [&]() -> int {
      if (state_->headless) {
        for (const auto &token : state_->pending) {
          if (token) {
            if (!token->WaitHost())
              return 1;
          }
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - state_->benchmark_start).count();
        std::cout << fmt::format("nbody: headless 1280x720, {} frames, {:.6f} s, {:.2f} FPS (3 in flight, 60 warmup)\n",
                                 GetFrameIndex(), seconds, GetFrameIndex() / seconds);
        std::cout << fmt::format(
            "nbody: {} particles, {} FLOP/frame, estimated {:.3f} TFLOP/s (20 FLOP/interaction, whole-frame time)\n",
            particle_count, flops_per_frame, flops_per_frame * (GetFrameIndex() / seconds) * 1e-12);
      }
      if (state_->velocity_download && !state_->velocity_download->WaitHost())
        return 1;
      if (!UpdateSpeeds())
        return 1;
      if (state_->speeds_ready) {
        std::cout << "nbody: downloaded " << particle_count << " particle velocities, max speed "
                  << state_->maximum_speed << ", 100 histogram bins\n";
        if (!state_->capture_directory.empty()) {
          std::error_code error;
          const auto path = std::filesystem::u8path(state_->capture_directory);
          std::filesystem::create_directories(path, error);
          if (error)
            return 1;
          std::ofstream file(path / "nbody_speeds.csv");
          file << "speed_min,speed_max,count,normalized\n";
          for (size_t i = 0; i < state_->speed_counts.size(); ++i)
            file << state_->maximum_speed * i / state_->speed_counts.size() << ','
                 << state_->maximum_speed * (i + 1) / state_->speed_counts.size() << ',' << state_->speed_counts[i]
                 << ',' << state_->normalized_speeds[i] << '\n';
          file.close();
          if (!file)
            return 1;
        }
      }
      if (!state_->capture_directory.empty()) {
        if (!state_->last_frame || !state_->last_frame->WaitHost())
          return 1;
        std::vector<uint8_t> pixels(state_->last_frame->GetSize());
        if (!state_->last_frame->CopyTo(pixels.data(), pixels.size()) ||
            !SaveImage(state_->capture_directory, "nbody", pixels, state_->last_frame_desc, state_->last_frame_space,
                       state_->last_frame_white_scale))
          return 1;
      }
      return 0;
    }();
  }
  state_.reset();
  return result;
}

bool NBodyModule::ResetParticles(grassland::rhi::ICommandContext *upload) {
  auto unit = [&]() { return std::uniform_real_distribution<float>()(state_->random); };
  auto sphere = [&]() {
    const float z = unit() * 2 - 1;
    const float radius = std::sqrt(1 - z * z);
    const float angle = unit() * 6.28318530718f;
    return glm::vec3(radius * std::sin(angle), radius * std::cos(angle), z) * std::cbrt(unit());
  };
  std::vector<glm::vec3> origins(state_->galaxy_count), speeds(state_->galaxy_count);
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
    auto galaxy = std::uniform_int_distribution<int>(0, state_->galaxy_count - 1)(state_->random);
    initial_positions[i] =
        sphere() * (2.0f * std::cbrt(10.0f / state_->galaxy_count)) + origins[galaxy] - mean_position;
    initial_velocities[i] = sphere() * 2.0f + speeds[galaxy] - mean_velocity;
  }
  return upload->CmdUploadBuffer(state_->positions.Get(), 0, initial_positions.data(), state_->bytes) &&
         upload->CmdUploadBuffer(state_->velocities.Get(), 0, initial_velocities.data(), state_->bytes);
}

bool NBodyModule::UpdateSpeeds() {
  if (!state_->velocity_download)
    return true;
  const auto status = state_->velocity_download->GetStatus();
  if (status == SUBMISSION_STATUS_PENDING)
    return true;
  if (status != SUBMISSION_STATUS_COMPLETED ||
      !state_->velocity_download->CopyTo(state_->downloaded_velocities.data(),
                                         state_->downloaded_velocities.size() * sizeof(glm::vec3)))
    return false;
  state_->velocity_download = {};
  state_->maximum_speed = 0;
  for (const auto &velocity : state_->downloaded_velocities) {
    const float speed = glm::length(velocity);
    if (!std::isfinite(speed)) {
      std::cerr << "nbody: non-finite particle speed\n";
      return false;
    }
    state_->maximum_speed = std::max(state_->maximum_speed, speed);
  }
  state_->speed_counts.fill(0);
  for (const auto &velocity : state_->downloaded_velocities) {
    const size_t bin =
        state_->maximum_speed > 0
            ? std::min(size_t(glm::length(velocity) / state_->maximum_speed * state_->speed_counts.size()),
                       state_->speed_counts.size() - 1)
            : 0;
    ++state_->speed_counts[bin];
  }
  const auto peak = *std::max_element(state_->speed_counts.begin(), state_->speed_counts.end());
  for (size_t i = 0; i < state_->speed_counts.size(); ++i)
    state_->normalized_speeds[i] = peak ? float(state_->speed_counts[i]) / peak : 0;
  state_->speeds_ready = true;
  return true;
}
