#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <thread>

#include "grassland/rhi/rhi.h"
#include "grassland/util/fps_counter.h"
#include "grassland/util/virtual_file_system.h"

namespace {
#include "built_in_shaders.inl"
}

int main(int argc, char **argv) {
  using namespace grassland::rhi;
  BackendAPI api = BACKEND_API_DEFAULT;
  bool headless = false;
  int frames = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      std::cout << "[--backend auto|d3d12|vulkan|metal] [--headless] [--frames N]\n"
                   "Headless defaults to 3000 measured frames after 60 warmup frames.\n"
                   "Window mode runs until closed; P toggles VSync.\n";
      return 0;
    }
    if (arg == "--headless") {
      headless = true;
      continue;
    }
    if (arg == "--frames" && i + 1 < argc) {
      const std::string value = argv[++i];
      const auto result = std::from_chars(value.data(), value.data() + value.size(), frames);
      if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || frames < 0) {
        std::cerr << "Invalid frame count: " << value << '\n';
        return 1;
      }
      continue;
    }
    if (arg == "--backend" && i + 1 < argc) {
      const std::string value = argv[++i];
      if (value == "auto")
        api = BACKEND_API_DEFAULT;
      else if (value == "d3d12")
        api = BACKEND_API_D3D12;
      else if (value == "vulkan")
        api = BACKEND_API_VULKAN;
      else if (value == "metal")
        api = BACKEND_API_METAL;
      else {
        std::cerr << "Unknown backend: " << value << '\n';
        return 1;
      }
      continue;
    }
    std::cerr << "Unknown or incomplete option: " << arg << '\n';
    return 1;
  }
  if (headless && !frames)
    frames = 3000;
  constexpr uint32_t body_count = 100000;
  ObjPtr<ICore> core = CreateCore(api);
  if (!core)
    return 1;
  DeviceFeatureRequirement required;
  required.ray_query = true;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection(required);
  if (!device) {
    std::cout << "tlas_pressure_test: SKIPPED (ray tracing unavailable)\n";
    return 0;
  }
  std::cout << "tlas_pressure_test: device " << device->Name() << "\n";
  ObjPtr<IWindow> window;
  if (!headless) {
    window = device->CreateGLFWWindow({1280, 720, "RHI tlas_pressure_test - P: toggle VSync"});
    if (!window)
      return 1;
  }
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue)
    return 1;
  std::array<ObjPtr<IImage>, 3> targets;
  std::array<ObjPtr<ISubmissionToken>, 3> pending;
  if (headless)
    for (auto &target : targets) {
      target = device->CreateImage({1280, 720, IMAGE_FORMAT_RGBA8_UNORM});
      if (!target)
        return 1;
    }
  const auto &embedded = shader_list.at("shaders/trace.slang");
  std::string source(embedded.first, embedded.second);
  const size_t prefix_offset = source.compare(0, 3, "\xef\xbb\xbf") == 0 ? 3 : 0;
  source.insert(prefix_offset, "#define PRESSURE_BODY_COUNT " + std::to_string(body_count) + "\n");
  PipelineLayoutDesc trace_layout{{{{{0, BINDING_ACCELERATION_STRUCTURE, 1, SHADER_VISIBILITY_COMPUTE},
                                     {1, BINDING_RW_TEXTURE2D, 1, SHADER_VISIBILITY_COMPUTE},
                                     {2, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                                     {3, BINDING_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_COMPUTE, 16}}}}};
  ObjPtr<IShader> cs = device->CreateShaderFromSource(source, "CSMain", SHADER_STAGE_COMPUTE);
  if (!cs)
    return 1;
  ObjPtr<IComputeProgram> trace = device->CreateComputeProgram({cs.Get(), trace_layout});
  if (!trace)
    return 1;

  const auto &display_source = shader_list.at("shaders/display.slang");
  std::string display(display_source.first, display_source.second);
  ObjPtr<IShader> vs = device->CreateShaderFromSource(display, "VSMain", SHADER_STAGE_VERTEX);
  if (!vs)
    return 1;
  ObjPtr<IShader> ps = device->CreateShaderFromSource(display, "PSMain", SHADER_STAGE_FRAGMENT);
  if (!ps)
    return 1;
  ProgramDesc graphics;
  graphics.vertex_shader = vs.Get();
  graphics.fragment_shader = ps.Get();
  graphics.pipeline_layout.groups = {{{{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT}}}};
  {
    ObjPtr<IImage> first;
    if (headless)
      first = targets[0];
    else
      first = window->GetNextImage();
    if (!first)
      return 1;
    graphics.color_formats = {first->GetDesc().format};
  }
  ObjPtr<IProgram> program = device->CreateProgram(graphics);
  if (!program)
    return 1;
  // One shared BLAS per shape. Motion changes instances, not mesh vertices.
  const float cube_vertices[] = {-.5f, -.5f, -.5f, .5f, -.5f, -.5f, .5f, .5f, -.5f, -.5f, .5f, -.5f,
                                 -.5f, -.5f, .5f,  .5f, -.5f, .5f,  .5f, .5f, .5f,  -.5f, .5f, .5f};
  const uint32_t cube_indices[] = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
                                   3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5};
  const float ground_vertices[] = {-240, 0, -240, 240, 0, -240, 240, 0, 240, -240, 0, -240, 240, 0, 240, -240, 0, 240};
  const float bounds[] = {-.5f, -.5f, -.5f, .5f, .5f, .5f};
  BottomLevelAccelerationStructureDesc cube_desc;
  cube_desc.geometries = {{8, 12, 36}};
  BottomLevelAccelerationStructureDesc sphere_desc;
  sphere_desc.aabbs = {{1, 24, true}};
  BottomLevelAccelerationStructureDesc ground_desc;
  ground_desc.geometries = {{6, 12, 0}};
  ObjPtr<IBottomLevelAccelerationStructure> cube = device->CreateBottomLevelAccelerationStructure(cube_desc);
  if (!cube)
    return 1;
  ObjPtr<IBottomLevelAccelerationStructure> sphere = device->CreateBottomLevelAccelerationStructure(sphere_desc);
  if (!sphere)
    return 1;
  ObjPtr<IBottomLevelAccelerationStructure> ground = device->CreateBottomLevelAccelerationStructure(ground_desc);
  if (!ground)
    return 1;

  struct Body {
    float x, y, z, velocity;
    bool sphere;
  };

  std::mt19937 random(87);
  std::uniform_real_distribution<float> position(-195.0f, 195.0f);
  std::uniform_real_distribution<float> color(.15f, .95f);
  std::uniform_real_distribution<float> bounce(3.0f, 10.0f);
  std::vector<Body> bodies(body_count);
  std::vector<std::array<float, 4>> colors(body_count + 1);
  for (size_t i = 0; i < bodies.size(); ++i) {
    bodies[i] = {position(random), .5f, position(random), 0, (random() & 1) != 0};
    colors[i] = {color(random), color(random), color(random), 1};
  }
  colors.back() = {.55f, .58f, .62f, 1};
  ObjPtr<IBuffer> color_buffer = device->CreateBuffer({colors.size() * sizeof(colors[0])});
  if (!color_buffer)
    return 1;
  {
    ObjPtr<IBuffer> vertices = device->CreateBuffer({sizeof(cube_vertices)});
    if (!vertices)
      return 1;
    ObjPtr<IBuffer> indices = device->CreateBuffer({sizeof(cube_indices)});
    if (!indices)
      return 1;
    ObjPtr<IBuffer> plane_vertices = device->CreateBuffer({sizeof(ground_vertices)});
    if (!plane_vertices)
      return 1;
    ObjPtr<IBuffer> aabbs = device->CreateBuffer({sizeof(bounds)});
    if (!aabbs)
      return 1;
    ObjPtr<ICommandContext> build = device->CreateCommandContext();
    if (!build)
      return 1;
    if (!build->CmdUploadBuffer(vertices.Get(), 0, cube_vertices, sizeof(cube_vertices)))
      return 1;
    if (!build->CmdUploadBuffer(indices.Get(), 0, cube_indices, sizeof(cube_indices)))
      return 1;
    if (!build->CmdUploadBuffer(plane_vertices.Get(), 0, ground_vertices, sizeof(ground_vertices)))
      return 1;
    if (!build->CmdUploadBuffer(aabbs.Get(), 0, bounds, sizeof(bounds)))
      return 1;
    if (!build->CmdUploadBuffer(color_buffer.Get(), 0, colors.data(), colors.size() * sizeof(colors[0])))
      return 1;
    if (!build->CmdBuildBottomLevelAccelerationStructure(cube.Get(), {{{vertices.Get()}, {indices.Get()}}}))
      return 1;
    if (!build->CmdBuildBottomLevelAccelerationStructure(sphere.Get(), {{{aabbs.Get()}, {}}}))
      return 1;
    if (!build->CmdBuildBottomLevelAccelerationStructure(ground.Get(), {{{plane_vertices.Get()}, {}}}))
      return 1;
    ObjPtr<ISubmissionToken> token = queue->Submit(build.Get());
    if (!token)
      return 1;
    if (!token->WaitHost())
      return 1;
  }
  ObjPtr<ITopLevelAccelerationStructure> tlas = device->CreateTopLevelAccelerationStructure({body_count + 1});
  if (!tlas)
    return 1;
  std::vector<uint32_t> visible, hidden(body_count);
  std::iota(hidden.begin(), hidden.end(), 0);
  visible.reserve(body_count);
  std::vector<RayTracingInstance> instances;
  instances.reserve(body_count + 1);
  double simulation_time = 0;
  auto previous_frame = std::chrono::steady_clock::now();

  struct Measurements {
    double simulation{}, build{}, render{}, submit{}, wait{};
    uint64_t frames{}, instances{};
  } totals;

  uint32_t minimum_visible = body_count, maximum_visible = 0;

  struct CameraInfo {
    glm::mat4 screen_to_camera;
    glm::mat4 camera_to_world;
  };

  ObjPtr<IBuffer> camera_buffer = device->CreateBuffer({256});
  if (!camera_buffer)
    return 1;
  ObjPtr<IBindingGroup> trace_group = device->CreateBindingGroup(trace_layout.groups[0]);
  if (!trace_group)
    return 1;
  if (!trace_group->BindResource(0, tlas.Get()))
    return 1;
  ObjPtr<IBindingGroup> display_group = device->CreateBindingGroup(graphics.pipeline_layout.groups[0]);
  if (!display_group)
    return 1;
  ObjPtr<IImage> output;
  if (!trace_group->BindResource(2, camera_buffer.Get()))
    return 1;
  if (!trace_group->BindResource(3, color_buffer.Get()))
    return 1;
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
    const auto wait_start = std::chrono::steady_clock::now();
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
    const auto wait_end = std::chrono::steady_clock::now();
    auto extent = image->GetDesc();
    ObjPtr<ICommandContext> context = device->CreateCommandContext();
    if (!context)
      return 1;
    if (!output || output->GetDesc().width != extent.width || output->GetDesc().height != extent.height) {
      output = device->CreateImage({extent.width, extent.height, IMAGE_FORMAT_RGBA8_UNORM});
      if (!output)
        return 1;
      if (!trace_group->BindResource(1, output.Get()))
        return 1;
      if (!display_group->BindResource(0, output.Get()))
        return 1;
      CameraInfo camera{glm::inverse(glm::perspective(glm::radians(55.0f), float(extent.width) / float(extent.height),
                                                      0.1f, 1500.0f)),
                        glm::inverse(glm::lookAt(glm::vec3{320, 320, 420}, glm::vec3{0, 1, 0}, glm::vec3{0, 1, 0}))};
      if (!context->CmdUploadBuffer(camera_buffer.Get(), 0, &camera, sizeof(camera)))
        return 1;
    }
    const auto simulation_start = std::chrono::steady_clock::now();
    double dt =
        headless ? 1.0 / 120.0 : std::min(.1, std::chrono::duration<double>(simulation_start - previous_frame).count());
    previous_frame = simulation_start;
    simulation_time += dt;
    // Simulate hidden bodies too, so showing one never resets its motion.
    while (dt > 0) {
      const double step = std::min(dt, 1.0 / 120.0);
      for (auto &body : bodies) {
        body.velocity -= 9.81f * float(step);
        body.y += body.velocity * float(step);
        if (body.y <= .5f) {
          body.y = .5f;
          body.velocity = bounce(random);
        }
      }
      dt -= step;
    }
    const uint32_t target_count =
        uint32_t(std::lround(body_count * (0.6 - 0.4 * std::cos(simulation_time * glm::two_pi<double>() / 8.0))));
    while (visible.size() != target_count) {
      auto &from = visible.size() < target_count ? hidden : visible;
      auto &to = visible.size() < target_count ? visible : hidden;
      const size_t index = std::uniform_int_distribution<size_t>(0, from.size() - 1)(random);
      to.push_back(from[index]);
      from[index] = from.back();
      from.pop_back();
    }
    instances.clear();
    RayTracingInstance plane;
    plane.blas = ground.Get();
    plane.custom_index = body_count;
    plane.disable_triangle_culling = true;
    instances.push_back(plane);
    for (uint32_t id : visible) {
      const auto &body = bodies[id];
      RayTracingInstance instance;
      instance.blas = body.sphere ? sphere.Get() : cube.Get();
      instance.custom_index = id;
      instance.disable_triangle_culling = true;
      instance.transform[3] = body.x;
      instance.transform[7] = body.y;
      instance.transform[11] = body.z;
      instances.push_back(instance);
    }
    const auto build_start = std::chrono::steady_clock::now();
    // A changing instance count requires BUILD, not UPDATE/refit. Storage is reused.
    if (!context->CmdBuildTopLevelAccelerationStructure(tlas.Get(), instances, ACCELERATION_STRUCTURE_BUILD))
      return 1;
    const auto render_start = std::chrono::steady_clock::now();
    if (!context->CmdBindComputeProgram(trace.Get()))
      return 1;
    if (!context->CmdBindGroup(0, trace_group.Get()))
      return 1;
    if (!context->CmdDispatch((extent.width + 7) / 8, (extent.height + 7) / 8))
      return 1;
    if (!context->CmdBindProgram(program.Get()))
      return 1;
    if (!context->CmdBindGroup(0, display_group.Get()))
      return 1;
    if (!context->CmdBeginRendering(
            {{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}))
      return 1;
    if (!context->CmdSetViewport({0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1}))
      return 1;
    if (!context->CmdSetScissor({0, 0, extent.width, extent.height}))
      return 1;
    if (!context->CmdDraw(3))
      return 1;
    if (!context->CmdEndRendering())
      return 1;
    if (!headless) {
      if (!context->CmdPresent(window.Get())) {
        // The OS can minimize the window during this long CPU recording interval.
        ObjPtr<IImage> next = window->GetNextImage();
        if (next)
          return 1;
        // Discard this recording; retry the camera upload if it was part of this frame.
        output.Release();
        fps.Reset();
        continue;
      }
    }
    const auto submit_start = std::chrono::steady_clock::now();
    ObjPtr<ISubmissionToken> token = queue->Submit(context.Get());
    if (!token)
      return 1;
    if (headless)
      pending[slot] = std::move(token);
    const auto submit_end = std::chrono::steady_clock::now();
    if (rendered >= 0) {
      auto milliseconds = [](auto start, auto end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
      };
      totals.simulation += milliseconds(simulation_start, build_start);
      totals.build += milliseconds(build_start, render_start);
      totals.render += milliseconds(render_start, submit_start);
      totals.submit += milliseconds(submit_start, submit_end);
      totals.wait += milliseconds(wait_start, wait_end);
      ++totals.frames;
      totals.instances += instances.size();
      minimum_visible = std::min(minimum_visible, target_count);
      maximum_visible = std::max(maximum_visible, target_count);
    }
    ++rendered;
    fps.TickFrame();
    auto now = std::chrono::steady_clock::now();
    if (!headless && now - title_time >= std::chrono::milliseconds(250)) {
      if (!window->SetTitle(
              fmt::format("RHI tlas_pressure_test - FPS: {:.1f} - Objects: {} - TLAS: {} - BUILD - P: VSync",
                          fps.GetFPS(), visible.size(), instances.size())))
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
    std::cout << fmt::format(
        "tlas_pressure_test: headless 1280x720, {} frames, {:.6f} s, {:.2f} FPS (3 in flight, 60 warmup)\n", rendered,
        seconds, rendered / seconds);
  }
  if (totals.frames) {
    const double n = double(totals.frames);
    std::cout << fmt::format(
        "tlas_pressure_test: visible {}..{}, average TLAS instances {:.1f}, CPU ms/frame: simulation {:.4f}, build recording {:.4f}, render recording {:.4f}, submit {:.4f}, frame wait {:.4f}\n",
        minimum_visible, maximum_visible, totals.instances / n, totals.simulation / n, totals.build / n,
        totals.render / n, totals.submit / n, totals.wait / n);
  }
  return 0;
}
