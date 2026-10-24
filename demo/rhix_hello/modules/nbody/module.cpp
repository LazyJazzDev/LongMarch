#include "../../module.h"

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
#include <map>
#include <random>
#include <string>
#include <thread>

#include "../../save_image.h"
#include "grassland/rhi/rhix/rhix.h"
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

int RunNBody(grassland::rhi::BackendAPI api, int frames, bool headless, const std::string &capture_directory) {
  using namespace grassland::rhi;

  grassland::rhix::Core core = grassland::rhix::CreateCore(api);
  if (!core)
    return 1;
  grassland::rhix::Device device = core.CreateDeviceByAutoSelection();
  if (!device)
    return 1;
  grassland::rhix::DownloadToken last_frame;
  ImageDesc last_frame_desc{};
  ColorSpace last_frame_space = COLOR_SPACE_SRGB_NONLINEAR;
  float last_frame_white_scale = 1.0f;
  PresentMode requested = PRESENT_MODE_VSYNC;
  bool present_mode_changed = false;
  bool running = true, reset_requested = true;
  float delta_t = 0.03f;
  int galaxy_count = 10;
  glm::mat4 rotation{1.0f};
  glm::dvec2 last_cursor{}, rotation_delta{};
  bool cursor_initialized = false, dragging = false;
  std::string status;
  bool capture_keyboard = false, output_change_requested = false;
  ColorSpace requested_space = COLOR_SPACE_SRGB_NONLINEAR;
  std::vector<ColorSpace> spaces{COLOR_SPACE_SRGB_NONLINEAR};
  grassland::rhix::Window window;
  if (!headless) {
    window = device.CreateGLFWWindow({1280, 720, "RHIX nbody"});
    if (!window)
      return 1;
    if (!window.EnableImGui(true, LONGMARCH_ASSETS_DIR "fonts/simhei.ttf", 20.0f))
      return 1;
    for (const auto &option : window.GetSupportedPresentationOptions()) {
      if (std::find(spaces.begin(), spaces.end(), option.color_space) == spaces.end())
        spaces.push_back(option.color_space);
    }
  }
  grassland::rhix::Queue queue = device.GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue)
    return 1;
  std::array<grassland::rhix::Image, 3> targets;
  std::array<grassland::rhix::SubmissionToken, 3> pending;
  if (headless) {
    for (auto &target : targets) {
      target = device.CreateImage({1280, 720, IMAGE_FORMAT_RGBA8_UNORM});
      if (!target)
        return 1;
    }
  }
  const auto &embedded = shader_list.at("modules/nbody/shaders/particle.slang");
  std::string source(embedded.first, embedded.second);
  grassland::rhix::Shader vertex_shader = device.CreateShaderFromSource(source, "VSMain", SHADER_STAGE_VERTEX);
  if (!vertex_shader)
    return 1;
  grassland::rhix::Shader fragment_shader = device.CreateShaderFromSource(source, "PSMain", SHADER_STAGE_FRAGMENT);
  if (!fragment_shader)
    return 1;
  ProgramDesc desc;
  desc.vertex_shader = vertex_shader.Get();
  desc.fragment_shader = fragment_shader.Get();
  desc.blend_states = {BlendState(BLEND_FACTOR_ONE, BLEND_FACTOR_ONE, BLEND_OP_ADD, BLEND_FACTOR_ONE,
                                  BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, BLEND_OP_ADD)};
  desc.vertex_bindings = {{sizeof(glm::vec3), true}};
  desc.vertex_attributes = {{0, 0, VERTEX_FORMAT_FLOAT3, 0}};
  desc.pipeline_layout.groups = {{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_VERTEX}}}};
  desc.color_formats = {IMAGE_FORMAT_RGBA16_FLOAT};
  grassland::rhix::Program program = device.CreateProgram(desc);
  if (!program)
    return 1;
  grassland::rhix::Image radiance;
  const auto &output_embedded = shader_list.at("modules/nbody/shaders/present.slang");
  const std::string output_source(output_embedded.first, output_embedded.second);
  const BindingLayoutDesc output_layout{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_FRAGMENT},
                                         {1, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT}}};
  grassland::rhix::Buffer output_uniform = device.CreateBuffer({256});
  grassland::rhix::BindingGroup output_group = device.CreateBindingGroup(output_layout);
  if (!output_uniform || !output_group || !output_group.BindResource(0, output_uniform))
    return 1;
  std::map<std::pair<ImageFormat, ColorSpace>, grassland::rhix::Program> output_programs;
  const auto &compute_source = shader_list.at("modules/nbody/shaders/nbody.slang");
  grassland::rhix::Shader compute_shader = device.CreateShaderFromSource(
      std::string(compute_source.first, compute_source.second), "CSMain", SHADER_STAGE_COMPUTE);
  if (!compute_shader)
    return 1;
  ComputeProgramDesc compute_desc;
  compute_desc.compute_shader = compute_shader.Get();
  compute_desc.pipeline_layout.groups = {{{{0, BINDING_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                                           {1, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                                           {2, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                                           {3, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_COMPUTE}}}};
  grassland::rhix::ComputeProgram compute = device.CreateComputeProgram(compute_desc);
  if (!compute)
    return 1;
  constexpr uint64_t bytes = particle_count * sizeof(glm::vec3);
  grassland::rhix::Buffer positions = device.CreateBuffer({bytes});
  if (!positions)
    return 1;
  grassland::rhix::Buffer velocities = device.CreateBuffer({bytes});
  if (!velocities)
    return 1;
  grassland::rhix::Buffer next_positions = device.CreateBuffer({bytes});
  if (!next_positions)
    return 1;
  grassland::rhix::Buffer simulation_uniform = device.CreateBuffer({256});
  if (!simulation_uniform)
    return 1;
  grassland::rhix::Buffer camera_uniform = device.CreateBuffer({256});
  if (!camera_uniform)
    return 1;
  grassland::rhix::BindingGroup simulation_group = device.CreateBindingGroup(compute_desc.pipeline_layout.groups[0]);
  if (!simulation_group)
    return 1;
  if (!simulation_group.BindResource(0, positions))
    return 1;
  if (!simulation_group.BindResource(1, velocities))
    return 1;
  if (!simulation_group.BindResource(2, next_positions))
    return 1;
  if (!simulation_group.BindResource(3, simulation_uniform))
    return 1;
  grassland::rhix::BindingGroup camera_group = device.CreateBindingGroup(desc.pipeline_layout.groups[0]);
  if (!camera_group)
    return 1;
  if (!camera_group.BindResource(0, camera_uniform))
    return 1;
  std::mt19937 random(42);
  auto reset_particles = [&](grassland::rhix::CommandContext &upload) {
    auto unit = [&]() { return std::uniform_real_distribution<float>()(random); };
    auto sphere = [&]() {
      const float z = unit() * 2 - 1;
      const float radius = std::sqrt(1 - z * z);
      const float angle = unit() * 6.28318530718f;
      return glm::vec3(radius * std::sin(angle), radius * std::cos(angle), z) * std::cbrt(unit());
    };
    std::vector<glm::vec3> origins(galaxy_count), speeds(galaxy_count);
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
      auto galaxy = std::uniform_int_distribution<int>(0, galaxy_count - 1)(random);
      initial_positions[i] = sphere() * (2.0f * std::cbrt(10.0f / galaxy_count)) + origins[galaxy] - mean_position;
      initial_velocities[i] = sphere() * 2.0f + speeds[galaxy] - mean_velocity;
    }
    return upload.CmdUploadBuffer(positions, 0, initial_positions.data(), bytes) &&
           upload.CmdUploadBuffer(velocities, 0, initial_velocities.data(), bytes);
  };
  if (!headless) {
    window.KeyEvent().RegisterCallback([&](int key, int, int action, int) {
      if (action != GLFW_PRESS)
        return;
      if (capture_keyboard && key != GLFW_KEY_ESCAPE)
        return;
      if (key == GLFW_KEY_SPACE)
        running = !running;
      if (key == GLFW_KEY_R)
        reset_requested = true;
      if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window.GLFWWindow(), GLFW_TRUE);
      if (key == GLFW_KEY_P) {
        requested = requested == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
        present_mode_changed = true;
      }
    });
  }
  if (window) {
    window.MouseButtonEvent().RegisterCallback([&](int button, int action, int, double, double) {
      if (button == GLFW_MOUSE_BUTTON_LEFT)
        dragging = action != GLFW_RELEASE;
    });
    window.MouseMoveEvent().RegisterCallback([&](double x, double y) {
      const glm::dvec2 cursor{x, y};
      if (cursor_initialized && dragging)
        rotation_delta += cursor - last_cursor;
      last_cursor = cursor;
      cursor_initialized = true;
    });
  }
  grassland::rhix::DownloadToken velocity_download;
  std::vector<glm::vec3> downloaded_velocities(particle_count);
  std::array<uint32_t, 100> speed_counts{};
  std::array<float, 100> normalized_speeds{};
  float maximum_speed = 0;
  bool speeds_ready = false;
  auto update_speeds = [&]() {
    if (!velocity_download)
      return true;
    const auto status = velocity_download.GetStatus();
    if (status == SUBMISSION_STATUS_PENDING)
      return true;
    if (status != SUBMISSION_STATUS_COMPLETED ||
        !velocity_download.CopyTo(downloaded_velocities.data(), downloaded_velocities.size() * sizeof(glm::vec3)))
      return false;
    velocity_download = {};
    maximum_speed = 0;
    for (const auto &velocity : downloaded_velocities) {
      const float speed = glm::length(velocity);
      if (!std::isfinite(speed)) {
        std::cerr << "nbody: non-finite particle speed\n";
        return false;
      }
      maximum_speed = std::max(maximum_speed, speed);
    }
    speed_counts.fill(0);
    for (const auto &velocity : downloaded_velocities) {
      const size_t bin =
          maximum_speed > 0
              ? std::min(size_t(glm::length(velocity) / maximum_speed * speed_counts.size()), speed_counts.size() - 1)
              : 0;
      ++speed_counts[bin];
    }
    const auto peak = *std::max_element(speed_counts.begin(), speed_counts.end());
    for (size_t i = 0; i < speed_counts.size(); ++i)
      normalized_speeds[i] = peak ? float(speed_counts[i]) / peak : 0;
    speeds_ready = true;
    return true;
  };
  grassland::FPSCounter fps;
  // Warm up allocations and pipelines before measuring completed GPU work.
  auto benchmark_start = std::chrono::steady_clock::now();
  int rendered = headless ? -60 : 0;
  while (!frames || rendered < frames) {
    if (!update_speeds())
      return 1;
    if (!headless) {
      if (!PollEvents())
        break;
      if (window.ShouldClose())
        break;
      if (!window.UpdateImGui([&] {
            ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Once);
            ImGui::SetNextWindowBgAlpha(0.3f);
            ImGui::SetNextWindowSize({ImGui::GetFontSize() * 30.0f, 0}, ImGuiCond_Always);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {8.0f, 6.0f});
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
            if (ImGui::Begin("NBody", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize)) {
              ImGui::SeparatorText("Statistics");
              ImGui::Text("Backend: %s", BackendAPIName(api).c_str());
              ImGui::Text("Particles: %u", particle_count);
              const double rate = fps.GetFPS();
              ImGui::Text("FPS: %.1f", rate);
              ImGui::Text("Frame Duration: %.3f ms", rate > 0 ? 1000.0 / rate : 0.0);
              ImGui::Text("Estimated: %.2f TFLOP/s", running ? flops_per_frame * rate * 1e-12 : 0.0);
              ImGui::SeparatorText("Control");
              const auto &style = ImGui::GetStyle();
              const float row_width = ImGui::GetContentRegionAvail().x;
              const ImVec2 button_size{(row_width - style.ItemSpacing.x) * 0.5f, ImGui::GetFrameHeight() + 6.0f};
              ImGui::PushStyleColor(ImGuiCol_Button, {0.20f, 0.25f, 0.32f, 1.0f});
              ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.28f, 0.35f, 0.44f, 1.0f});
              ImGui::PushStyleColor(ImGuiCol_ButtonActive, {0.33f, 0.41f, 0.52f, 1.0f});
              if (ImGui::Button("Reset", button_size))
                reset_requested = true;
              ImGui::PopStyleColor(3);
              ImGui::SameLine();
              ImGui::PushStyleColor(ImGuiCol_Button,
                                    running ? ImVec4{0.16f, 0.34f, 0.52f, 1.0f} : ImVec4{0.14f, 0.38f, 0.28f, 1.0f});
              ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                    running ? ImVec4{0.20f, 0.43f, 0.65f, 1.0f} : ImVec4{0.18f, 0.49f, 0.36f, 1.0f});
              ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                    running ? ImVec4{0.13f, 0.28f, 0.43f, 1.0f} : ImVec4{0.10f, 0.30f, 0.21f, 1.0f});
              if (ImGui::Button(running ? "Pause" : "Resume", button_size))
                running = !running;
              ImGui::PopStyleColor(3);
              const float label_width =
                  std::max(ImGui::CalcTextSize("Galaxy Number").x,
                           ImGui::GetFrameHeight() + style.ItemInnerSpacing.x + ImGui::CalcTextSize("VSync").x);
              const float field_width = row_width - label_width - style.ItemSpacing.x;
              ImGui::SetNextItemWidth(field_width);
              ImGui::SliderFloat("##DeltaTime", &delta_t, 0.001f, 0.1f, "%.3f", ImGuiSliderFlags_Logarithmic);
              ImGui::SameLine();
              ImGui::TextUnformatted("Delta Time");
              ImGui::SetNextItemWidth(field_width);
              ImGui::SliderInt("##GalaxyNumber", &galaxy_count, 1, 20);
              ImGui::SameLine();
              ImGui::TextUnformatted("Galaxy Number");
              ImGui::SeparatorText("ColorSpace");
              ImGui::SetNextItemWidth(field_width);
              if (ImGui::BeginCombo("##ColorSpace", ColorSpaceName(window.GetColorSpace()).c_str())) {
                for (auto space : spaces) {
                  if (ImGui::Selectable(ColorSpaceName(space).c_str(), space == window.GetColorSpace())) {
                    requested_space = space;
                    output_change_requested = true;
                  }
                }
                ImGui::EndCombo();
              }
              ImGui::SameLine();
              bool vsync = requested == PRESENT_MODE_VSYNC;
              if (ImGui::Checkbox("VSync", &vsync)) {
                requested = vsync ? PRESENT_MODE_VSYNC : PRESENT_MODE_CONTINUOUS;
                present_mode_changed = true;
              }
              if (!status.empty())
                ImGui::TextUnformatted(status.c_str());
              ImGui::SeparatorText("Speed Distribution");
              if (speeds_ready) {
                ImGui::PlotLines("##Speeds", normalized_speeds.data(), int(normalized_speeds.size()), 0, nullptr, 0, 1,
                                 {ImGui::GetContentRegionAvail().x, 80});
                ImGui::Text("Speed: 0 - %.3f", maximum_speed);
              } else
                ImGui::TextUnformatted("Waiting for GPU readback...");
              ImGui::SeparatorText("Instructions");
              ImGui::TextWrapped("Galaxy count applies on Reset. Drag outside UI to rotate.");
              ImGui::TextWrapped("P: VSync; Space: pause/resume; R: reset; Escape: close");
            }
            ImGui::End();
            ImGui::PopStyleVar(2);
            capture_keyboard = ImGui::GetIO().WantCaptureKeyboard;
            if (!ImGui::GetIO().WantCaptureMouse) {
              rotation =
                  glm::rotate(glm::mat4{1}, glm::radians(float(rotation_delta.x)), glm::vec3{0, 1, 0}) * rotation;
              rotation =
                  glm::rotate(glm::mat4{1}, glm::radians(float(rotation_delta.y)), glm::vec3{1, 0, 0}) * rotation;
            }
            rotation_delta = {};
          }))
        return 1;
      if (output_change_requested) {
        const auto next = requested_space;
        output_change_requested = false;
        if (next == window.GetColorSpace())
          status.clear();
        else if (!window.SetColorSpace(next))
          status = "HDR mode unavailable; keeping the current output.";
        else {
          status.clear();
          fps.Reset();
          std::cout << "nbody: output " << ColorSpaceName(next) << '\n';
        }
      }
      if (window.IsSwapChainInvalid()) {
        if (!window.RebuildSwapChain())
          return 1;
      }
      if (present_mode_changed) {
        present_mode_changed = false;
        if (!window.SetPresentMode(requested))
          return 1;
      }
    }
    const size_t slot = (static_cast<int64_t>(rendered) + 60) % targets.size();
    if (headless) {
      if (rendered == 0) {
        for (auto &token : pending) {
          if (token) {
            if (!token.WaitHost())
              return 1;
            token = {};
          }
        }
        benchmark_start = std::chrono::steady_clock::now();
      }
      if (pending[slot]) {
        if (!pending[slot].WaitHost())
          return 1;
        pending[slot] = {};
      }
    }
    grassland::rhix::Image image;
    if (headless)
      image = targets[slot];
    else
      image = window.GetNextImage();
    if (!image) {
      if (headless)
        return 1;
      fps.Reset();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    auto extent = image.GetDesc();
    if (!radiance || radiance.GetDesc().width != extent.width || radiance.GetDesc().height != extent.height) {
      radiance = device.CreateImage({extent.width, extent.height, IMAGE_FORMAT_RGBA16_FLOAT});
      if (!radiance || !output_group.BindResource(1, radiance))
        return 1;
    }
    const auto space = headless ? COLOR_SPACE_SRGB_NONLINEAR : window.GetColorSpace();
    auto &output_program = output_programs[{extent.format, space}];
    if (!output_program) {
      const auto code = "#define OUTPUT_COLOR_SPACE " + std::to_string(int(space)) + "\n" + output_source;
      grassland::rhix::Shader vs = device.CreateShaderFromSource(code, "VSMain", SHADER_STAGE_VERTEX);
      grassland::rhix::Shader ps = device.CreateShaderFromSource(code, "PSMain", SHADER_STAGE_FRAGMENT);
      if (!vs || !ps)
        return 1;
      ProgramDesc output_desc;
      output_desc.vertex_shader = vs.Get();
      output_desc.fragment_shader = ps.Get();
      output_desc.color_formats = {extent.format};
      output_desc.pipeline_layout.groups = {output_layout};
      output_program = device.CreateProgram(output_desc);
      if (!output_program)
        return 1;
    }
    grassland::rhix::CommandContext context = device.CreateCommandContext();
    if (!context)
      return 1;
    const auto view = glm::lookAt(glm::vec3(10, 20, 30), glm::vec3(0), glm::vec3(0, 1, 0)) * rotation;
    Camera camera{glm::perspective(glm::radians(60.0f), float(extent.width) / extent.height, 0.1f, 100.0f) * view,
                  glm::inverse(view), 0.07f};
    if (!context.CmdUploadBuffer(camera_uniform, 0, &camera, sizeof(camera)))
      return 1;
    if (reset_requested) {
      if (!reset_particles(context))
        return 1;
      reset_requested = false;
      std::cout << "nbody: reset " << galaxy_count << " galaxies\n";
    }
    if (running) {
      const Simulation settings{particle_count, delta_t, 100.0f / particle_count};
      if (!context.CmdUploadBuffer(simulation_uniform, 0, &settings, sizeof(settings)) ||
          !context.CmdBindComputeProgram(compute) || !context.CmdBindGroup(0, simulation_group) ||
          !context.CmdDispatch(particle_count / 128) || !context.CmdCopyBuffer(next_positions, 0, positions, 0, bytes))
        return 1;
    }
    if (!context.CmdBindProgram(program))
      return 1;
    if (!context.CmdBindGroup(0, camera_group))
      return 1;
    if (!context.CmdBeginRendering(radiance, ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}))
      return 1;
    if (!context.CmdBindVertexBuffer(0, positions))
      return 1;
    if (!context.CmdSetViewport({0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1}))
      return 1;
    if (!context.CmdSetScissor({0, 0, extent.width, extent.height}))
      return 1;
    if (!context.CmdDraw(6, particle_count))
      return 1;
    if (!context.CmdEndRendering())
      return 1;
    const float display[4] = {headless ? 1.0f : window.GetHDRReferenceWhiteScale(),
                              headless ? 1.0f : window.GetHDRHeadroom(), 0, 0};
    if (!context.CmdUploadBuffer(output_uniform, 0, display, sizeof(display)) ||
        !context.CmdBindProgram(output_program) || !context.CmdBindGroup(0, output_group) ||
        !context.CmdBeginRendering(image, ATTACHMENT_LOAD_OP_DISCARD, ATTACHMENT_STORE_OP_STORE) ||
        !context.CmdSetViewport({0, 0, float(extent.width), float(extent.height), 0, 1}) ||
        !context.CmdSetScissor({0, 0, extent.width, extent.height}) || !context.CmdDraw(3) ||
        !context.CmdEndRendering())
      return 1;
    if (!headless) {
      if (!context.CmdPresent(window))
        return 1;
    }
    if (!velocity_download) {
      velocity_download = context.CmdDownloadBuffer(velocities, 0, particle_count * sizeof(glm::vec3));
      if (!velocity_download)
        return 1;
    }
    if (!capture_directory.empty()) {
      last_frame = context.CmdDownloadImage(image);
      if (!last_frame)
        return 1;
      last_frame_desc = image.GetDesc();
      last_frame_space = window ? window.GetColorSpace() : COLOR_SPACE_SRGB_NONLINEAR;
      last_frame_white_scale = window ? window.GetHDRReferenceWhiteScale() : 1.0f;
    }
    grassland::rhix::SubmissionToken token = queue.Submit(std::move(context));

    if (!token)
      return 1;
    if (headless)
      pending[slot] = std::move(token);
    ++rendered;
    fps.TickFrame();
  }
  if (headless) {
    for (const auto &token : pending) {
      if (token) {
        if (!token.WaitHost())
          return 1;
      }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - benchmark_start).count();
    std::cout << fmt::format("nbody: headless 1280x720, {} frames, {:.6f} s, {:.2f} FPS (3 in flight, 60 warmup)\n",
                             rendered, seconds, rendered / seconds);
    std::cout << fmt::format(
        "nbody: {} particles, {} FLOP/frame, estimated {:.3f} TFLOP/s (20 FLOP/interaction, whole-frame time)\n",
        particle_count, flops_per_frame, flops_per_frame * (rendered / seconds) * 1e-12);
  }
  if (velocity_download && !velocity_download.WaitHost())
    return 1;
  if (!update_speeds())
    return 1;
  if (speeds_ready) {
    std::cout << "nbody: downloaded " << particle_count << " particle velocities, max speed " << maximum_speed
              << ", 100 histogram bins\n";
    if (!capture_directory.empty()) {
      std::error_code error;
      const auto path = std::filesystem::u8path(capture_directory);
      std::filesystem::create_directories(path, error);
      if (error)
        return 1;
      std::ofstream file(path / "nbody_speeds.csv");
      file << "speed_min,speed_max,count,normalized\n";
      for (size_t i = 0; i < speed_counts.size(); ++i)
        file << maximum_speed * i / speed_counts.size() << ',' << maximum_speed * (i + 1) / speed_counts.size() << ','
             << speed_counts[i] << ',' << normalized_speeds[i] << '\n';
      file.close();
      if (!file)
        return 1;
    }
  }
  if (!capture_directory.empty()) {
    if (!last_frame || !last_frame.WaitHost())
      return 1;
    std::vector<uint8_t> pixels(last_frame.GetSize());
    if (!last_frame.CopyTo(pixels.data(), pixels.size()) ||
        !SaveImage(capture_directory, "nbody", pixels, last_frame_desc, last_frame_space, last_frame_white_scale))
      return 1;
  }
  return 0;
}
