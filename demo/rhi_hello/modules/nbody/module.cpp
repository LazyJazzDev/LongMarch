#include "../../module.h"

#include <fmt/format.h>

#include <array>
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <thread>

#include "grassland/rhi/rhi.h"
#include "grassland/util/fps_counter.h"
#include "grassland/util/virtual_file_system.h"

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

int RunNBody(grassland::rhi::BackendAPI api, int frames, bool headless) {
  using namespace grassland::rhi;

  ObjPtr<ICore> core = CreateCore(api);
  if (!core)
    return 1;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  if (!device)
    return 1;
  ObjPtr<IWindow> window;
  if (!headless) {
    window = device->CreateGLFWWindow({1280, 720, "RHI nbody - P: toggle VSync"});
    if (!window)
      return 1;
  }
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue)
    return 1;
  std::array<ObjPtr<IImage>, 3> targets;
  std::array<ObjPtr<ISubmissionToken>, 3> pending;
  if (headless) {
    for (auto &target : targets) {
      target = device->CreateImage({1280, 720, IMAGE_FORMAT_RGBA8_UNORM});
      if (!target)
        return 1;
    }
  }
  const auto &embedded = shader_list.at("modules/nbody/shaders/particle.slang");
  std::string source(embedded.first, embedded.second);
  ObjPtr<IShader> vertex_shader = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  if (!vertex_shader)
    return 1;
  ObjPtr<IShader> fragment_shader = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
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
  {
    ObjPtr<IImage> image;
    if (headless)
      image = targets[0];
    else
      image = window->GetNextImage();
    if (!image)
      return 1;
    desc.color_formats = {image->GetDesc().format};
  }
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  if (!program)
    return 1;
  if (!program->Finalize())
    return 1;
  const auto &compute_source = shader_list.at("modules/nbody/shaders/nbody.slang");
  ObjPtr<IShader> compute_shader =
      device->CreateShader({std::string(compute_source.first, compute_source.second), "CSMain", SHADER_STAGE_COMPUTE});
  if (!compute_shader)
    return 1;
  ComputeProgramDesc compute_desc;
  compute_desc.compute_shader = compute_shader.Get();
  compute_desc.pipeline_layout.groups = {{{{0, BINDING_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                                           {1, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                                           {2, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                                           {3, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_COMPUTE}}}};
  ObjPtr<IComputeProgram> compute = device->CreateComputeProgram(compute_desc);
  if (!compute)
    return 1;
  if (!compute->Finalize())
    return 1;
  constexpr uint64_t bytes = particle_count * sizeof(glm::vec3);
  ObjPtr<IBuffer> positions = device->CreateBuffer({bytes});
  if (!positions)
    return 1;
  ObjPtr<IBuffer> velocities = device->CreateBuffer({bytes});
  if (!velocities)
    return 1;
  ObjPtr<IBuffer> next_positions = device->CreateBuffer({bytes});
  if (!next_positions)
    return 1;
  ObjPtr<IBuffer> simulation_uniform = device->CreateBuffer({256});
  if (!simulation_uniform)
    return 1;
  ObjPtr<IBuffer> camera_uniform = device->CreateBuffer({256});
  if (!camera_uniform)
    return 1;
  ObjPtr<IBindingGroup> simulation_group = device->CreateBindingGroup(compute_desc.pipeline_layout.groups[0]);
  if (!simulation_group)
    return 1;
  if (!simulation_group->BindResource(0, positions.Get()))
    return 1;
  if (!simulation_group->BindResource(1, velocities.Get()))
    return 1;
  if (!simulation_group->BindResource(2, next_positions.Get()))
    return 1;
  if (!simulation_group->BindResource(3, simulation_uniform.Get()))
    return 1;
  ObjPtr<IBindingGroup> camera_group = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  if (!camera_group)
    return 1;
  if (!camera_group->BindResource(0, camera_uniform.Get()))
    return 1;
  {
    // Same ten-galaxy initialization and physical constants as nbody_cs.
    std::mt19937 random(42);
    auto unit = [&]() { return std::uniform_real_distribution<float>()(random); };
    auto sphere = [&]() {
      const float z = unit() * 2 - 1;
      const float radius = std::sqrt(1 - z * z);
      const float angle = unit() * 6.28318530718f;
      return glm::vec3(radius * std::sin(angle), radius * std::cos(angle), z) * std::cbrt(unit());
    };
    std::array<glm::vec3, 10> origins, speeds;
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
      auto galaxy = std::uniform_int_distribution<int>(0, 9)(random);
      initial_positions[i] = sphere() * 2.0f + origins[galaxy] - mean_position;
      initial_velocities[i] = sphere() * 2.0f + speeds[galaxy] - mean_velocity;
    }
    Simulation settings{particle_count, 0.03f, 100.0f / particle_count};
    ObjPtr<ICommandContext> upload = device->CreateCommandContext();
    if (!upload)
      return 1;
    if (!upload->CmdUploadBuffer(positions.Get(), 0, initial_positions.data(), bytes))
      return 1;
    if (!upload->CmdUploadBuffer(velocities.Get(), 0, initial_velocities.data(), bytes))
      return 1;
    if (!upload->CmdUploadBuffer(simulation_uniform.Get(), 0, &settings, sizeof(settings)))
      return 1;
    ObjPtr<ISubmissionToken> uploaded = queue->Submit(upload.Get());
    if (!uploaded)
      return 1;
    if (!uploaded->WaitHost())
      return 1;
  }
  bool previous_p = false;
  PresentMode requested = PRESENT_MODE_VSYNC;
  grassland::FPSCounter fps;
  auto title_time = std::chrono::steady_clock::now();
  // Warm up allocations and pipelines before measuring completed GPU work.
  auto benchmark_start = std::chrono::steady_clock::now();
  int rendered = headless ? -60 : 0;
  while (!frames || rendered < frames) {
    if (!headless) {
      if (!PollEvents())
        break;
      if (window->ShouldClose())
        break;
      if (window->IsKeyDown(GLFW_KEY_ESCAPE))
        break;
      if (window->IsSwapChainInvalid()) {
        if (!window->RebuildSwapChain())
          return 1;
      }
      bool p = window->IsKeyDown(GLFW_KEY_P);
      if (p && !previous_p) {
        requested = requested == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
        if (!window->SetPresentMode(requested))
          return 1;
      }
      previous_p = p;
    }
    const size_t slot = (static_cast<int64_t>(rendered) + 60) % targets.size();
    if (headless) {
      if (rendered == 0) {
        for (auto &token : pending) {
          if (token) {
            if (!token->WaitHost())
              return 1;
            token = {};
          }
        }
        benchmark_start = std::chrono::steady_clock::now();
      }
      if (pending[slot]) {
        if (!pending[slot]->WaitHost())
          return 1;
        pending[slot] = {};
      }
    }
    ObjPtr<IImage> image;
    if (headless)
      image = targets[slot];
    else
      image = window->GetNextImage();
    if (!image) {
      if (headless)
        return 1;
      fps.Reset();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    auto extent = image->GetDesc();
    ObjPtr<ICommandContext> context = device->CreateCommandContext();
    if (!context)
      return 1;
    const auto view = glm::lookAt(glm::vec3(10, 20, 30), glm::vec3(0), glm::vec3(0, 1, 0));
    Camera camera{glm::perspective(glm::radians(60.0f), float(extent.width) / extent.height, 0.1f, 100.0f) * view,
                  glm::inverse(view), 0.07f};
    if (!context->CmdUploadBuffer(camera_uniform.Get(), 0, &camera, sizeof(camera)))
      return 1;
    if (!context->CmdBindComputeProgram(compute.Get()))
      return 1;
    if (!context->CmdBindGroup(0, simulation_group.Get()))
      return 1;
    if (!context->CmdDispatch(particle_count / 128))
      return 1;
    if (!context->CmdCopyBuffer(next_positions.Get(), 0, positions.Get(), 0, bytes))
      return 1;
    if (!context->CmdBindProgram(program.Get()))
      return 1;
    if (!context->CmdBindGroup(0, camera_group.Get()))
      return 1;
    if (!context->CmdBeginRendering(
            {{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}))
      return 1;
    if (!context->CmdBindVertexBuffer(0, positions.Get()))
      return 1;
    if (!context->CmdSetViewport({0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1}))
      return 1;
    if (!context->CmdSetScissor({0, 0, extent.width, extent.height}))
      return 1;
    if (!context->CmdDraw(6, particle_count))
      return 1;
    if (!context->CmdEndRendering())
      return 1;
    if (!headless) {
      if (!context->CmdPresent(window.Get()))
        return 1;
    }
    ObjPtr<ISubmissionToken> token = queue->Submit(context.Get());
    if (!token)
      return 1;
    if (headless)
      pending[slot] = std::move(token);
    ++rendered;
    fps.TickFrame();
    auto now = std::chrono::steady_clock::now();
    if (!headless && now - title_time >= std::chrono::milliseconds(250)) {
      const double frame_rate = fps.GetFPS();
      if (!window->SetTitle(fmt::format("RHI nbody - FPS: {:.1f} - est. {:.2f} TFLOP/s - P: toggle VSync", frame_rate,
                                        flops_per_frame * frame_rate * 1e-12)))
        return 1;
      title_time = now;
    }
  }
  if (headless) {
    for (const auto &token : pending) {
      if (token) {
        if (!token->WaitHost())
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
  return 0;
}
