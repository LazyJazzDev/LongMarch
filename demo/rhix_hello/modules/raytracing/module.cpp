#include "../../module.h"

#include <fmt/format.h>

#include <array>
#include <chrono>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <map>
#include <thread>

#include "../../../rhi_common/last_frame.h"
#include "grassland/rhi/rhix/rhix.h"
#include "grassland/util/fps_counter.h"
#include "grassland/util/virtual_file_system.h"

namespace {
#include "built_in_shaders.inl"
}

int RunRayTracing(grassland::rhi::BackendAPI api, int frames, bool headless, const std::string &capture_directory) {
  using namespace grassland::rhi;
  using namespace grassland::rhix;
  Core core = grassland::rhix::CreateCore(api);
  if (!core)
    return 1;
  DeviceFeatureRequirement required;
  required.ray_tracing_pipeline = true;
  Device device = core.CreateDeviceByAutoSelection(required);
  if (!device) {
    std::cout << "raytracing: SKIPPED (ray tracing unavailable)\n";
    return 0;
  }
  rhi_demo::LastFrame capture(capture_directory, "raytracing");
  PresentMode requested = PRESENT_MODE_VSYNC;
  bool present_mode_changed = false;
  Window window;
  if (!headless) {
    window = device.CreateGLFWWindow({1280, 720, "RHIX raytracing - P: toggle VSync"});
    if (!window)
      return 1;
  }
  Queue queue = device.GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue)
    return 1;
  std::array<Image, 3> targets;
  std::array<SubmissionToken, 3> pending;
  if (headless)
    for (auto &target : targets) {
      target = device.CreateImage({1280, 720, IMAGE_FORMAT_RGBA8_UNORM});
      if (!target)
        return 1;
    }
  const auto &embedded = shader_list.at("modules/raytracing/shaders/trace.slang");
  std::string source(embedded.first, embedded.second);
  PipelineLayoutDesc trace_layout{{{{{0, BINDING_ACCELERATION_STRUCTURE, 1, SHADER_VISIBILITY_RAY_GENERATION},
                                     {1, BINDING_RW_TEXTURE2D, 1, SHADER_VISIBILITY_RAY_GENERATION},
                                     {2, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_RAY_GENERATION}}}}};
  Shader raygen = device.CreateShaderFromSource(source, "RayGenMain", SHADER_STAGE_RAY_GENERATION);
  if (!raygen)
    return 1;
  Shader miss = device.CreateShaderFromSource(source, "MissMain", SHADER_STAGE_MISS);
  if (!miss)
    return 1;
  Shader hit = device.CreateShaderFromSource(source, "ClosestHitMain", SHADER_STAGE_CLOSEST_HIT);
  if (!hit)
    return 1;
  Shader sphere_hit = device.CreateShaderFromSource(source, "SphereClosestHitMain", SHADER_STAGE_CLOSEST_HIT);
  if (!sphere_hit)
    return 1;
  Shader intersection = device.CreateShaderFromSource(source, "SphereIntersectionMain", SHADER_STAGE_INTERSECTION);
  if (!intersection)
    return 1;
  Shader callable = device.CreateShaderFromSource(source, "CallableMain", SHADER_STAGE_CALLABLE);
  if (!callable)
    return 1;
  RayTracingProgramDesc desc;
  desc.ray_generation = raygen.Get();
  desc.miss_shaders = {miss.Get()};
  desc.hit_groups = {{hit.Get(), nullptr, nullptr}, {sphere_hit.Get(), nullptr, intersection.Get()}};
  desc.pipeline_layout = trace_layout;
  desc.callable_shaders = {callable.Get()};
  desc.max_payload_size = 32;
  desc.max_attribute_size = 12;
  RayTracingProgram trace = device.CreateRayTracingProgram(desc);
  if (!trace)
    return 1;

  const auto &display_source = shader_list.at("modules/raytracing/shaders/display.slang");
  std::string display(display_source.first, display_source.second);
  Shader vs = device.CreateShaderFromSource(display, "VSMain", SHADER_STAGE_VERTEX);
  if (!vs)
    return 1;
  Shader ps = device.CreateShaderFromSource(display, "PSMain", SHADER_STAGE_FRAGMENT);
  if (!ps)
    return 1;
  ProgramDesc graphics;
  graphics.vertex_shader = vs.Get();
  graphics.fragment_shader = ps.Get();
  graphics.pipeline_layout.groups = {{{{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT}}}};
  {
    Image first;
    if (headless)
      first = targets[0];
    else
      first = window.GetNextImage();
    if (!first)
      return 1;
    graphics.color_formats = {first.GetDesc().format};
  }
  Program program = device.CreateProgram(graphics);
  if (!program)
    return 1;
  BottomLevelAccelerationStructureDesc triangles;
  triangles.geometries = {{3, 12, 0}};
  triangles.allow_compaction = true;
  BottomLevelAccelerationStructureDesc boxes;
  boxes.aabbs = {{1, 24, true}};
  BottomLevelAccelerationStructure triangle = device.CreateBottomLevelAccelerationStructure(triangles);
  if (!triangle)
    return 1;
  BottomLevelAccelerationStructure sphere = device.CreateBottomLevelAccelerationStructure(boxes);
  if (!sphere)
    return 1;
  TopLevelAccelerationStructure tlas = device.CreateTopLevelAccelerationStructure({2, true});
  if (!tlas)
    return 1;
  {
    const float vertices[] = {-1, -1, 0, 1, -1, 0, 0, 1, 0};
    const float bounds[] = {-1, -1, -1, 1, 1, 1};
    Buffer vertex_buffer = device.CreateBuffer({sizeof(vertices)});
    if (!vertex_buffer)
      return 1;
    Buffer aabb_buffer = device.CreateBuffer({sizeof(bounds)});
    if (!aabb_buffer)
      return 1;
    CommandContext build = device.CreateCommandContext();
    if (!build)
      return 1;
    if (!build.CmdUploadBuffer(vertex_buffer, 0, vertices, sizeof(vertices)))
      return 1;
    if (!build.CmdUploadBuffer(aabb_buffer, 0, bounds, sizeof(bounds)))
      return 1;
    if (!build.CmdBuildBottomLevelAccelerationStructure(triangle, {{{vertex_buffer}, {}}}))
      return 1;
    if (!build.CmdBuildBottomLevelAccelerationStructure(sphere, {{{aabb_buffer}, {}}}))
      return 1;
    if (!build.CmdQueryCompactedSize(triangle))
      return 1;
    SubmissionToken token = queue.Submit(std::move(build));
    if (!token)
      return 1;
    if (!token.WaitHost())
      return 1;
    triangles.compacted_size = triangle.GetCompactedSize();
    if (!triangles.compacted_size)
      return 1;
    BottomLevelAccelerationStructure compact = device.CreateBottomLevelAccelerationStructure(triangles);
    if (!compact)
      return 1;
    CommandContext copy = device.CreateCommandContext();
    if (!copy)
      return 1;
    if (!copy.CmdCompactAccelerationStructure(triangle, compact))
      return 1;
    token = queue.Submit(std::move(copy));
    if (!token)
      return 1;
    if (!token.WaitHost())
      return 1;
    triangle = std::move(compact);
  }
  std::vector<grassland::rhix::RayTracingInstance> instances(2);
  instances[0].blas = triangle;
  instances[1].blas = sphere;
  instances[1].transform[3] = 2.0f;
  instances[1].hit_group_offset = 1;
  {
    CommandContext build = device.CreateCommandContext();
    if (!build)
      return 1;
    if (!build.CmdBuildTopLevelAccelerationStructure(tlas, instances))
      return 1;
    SubmissionToken token = queue.Submit(std::move(build));
    if (!token)
      return 1;
    if (!token.WaitHost())
      return 1;
  }

  struct CameraInfo {
    glm::mat4 screen_to_camera;
    glm::mat4 camera_to_world;
  };

  Buffer camera_buffer = device.CreateBuffer({256});
  if (!camera_buffer)
    return 1;
  BindingGroup trace_group = device.CreateBindingGroup(trace_layout.groups[0]);
  if (!trace_group)
    return 1;
  if (!trace_group.BindResource(0, tlas))
    return 1;
  BindingGroup display_group = device.CreateBindingGroup(graphics.pipeline_layout.groups[0]);
  if (!display_group)
    return 1;
  Image output;
  if (!trace_group.BindResource(2, camera_buffer))
    return 1;
  if (!headless) {
    window.KeyEvent().RegisterCallback([&](int key, int, int action, int) {
      if (action != GLFW_PRESS)
        return;
      if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window.GLFWWindow(), GLFW_TRUE);
      if (key == GLFW_KEY_P) {
        requested = requested == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
        present_mode_changed = true;
      }
    });
  }
  grassland::FPSCounter fps;
  auto title_time = std::chrono::steady_clock::now();
  // Warm up allocations and pipelines before measuring completed GPU work.
  auto benchmark_start = std::chrono::steady_clock::now();
  const auto animation_start = std::chrono::steady_clock::now();
  int rendered = headless ? -60 : 0;
  while (!frames || rendered < frames) {
    if (!headless) {
      if (!PollEvents())
        break;
      if (window.ShouldClose())
        break;
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
    Image image;
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
    CommandContext context = device.CreateCommandContext();
    if (!context)
      return 1;
    if (!output || output.GetDesc().width != extent.width || output.GetDesc().height != extent.height) {
      output = device.CreateImage({extent.width, extent.height, IMAGE_FORMAT_RGBA8_UNORM});
      if (!output)
        return 1;
      if (!trace_group.BindResource(1, output))
        return 1;
      if (!display_group.BindResource(0, output))
        return 1;
      CameraInfo camera{
          glm::inverse(glm::perspective(glm::radians(60.0f), float(extent.width) / float(extent.height), 0.1f, 10.0f)),
          glm::inverse(glm::lookAt(glm::vec3{0, 0, 5}, glm::vec3{0, 0, 0}, glm::vec3{0, 1, 0}))};
      if (!context.CmdUploadBuffer(camera_buffer, 0, &camera, sizeof(camera)))
        return 1;
    }
    // Match graphics_hello: one Y-axis revolution every two seconds.
    const float angle = headless
                            ? float(rendered + 60) * 0.01f
                            : std::chrono::duration<float>(std::chrono::steady_clock::now() - animation_start).count() *
                                  glm::pi<float>();
    instances[0].transform = {std::cos(angle),  0, std::sin(angle), -2, 0, 1, 0, 0,
                              -std::sin(angle), 0, std::cos(angle), 0};
    instances[1].transform = {std::cos(angle),  0, std::sin(angle) * 0.5f, 2, 0, 1, 0, 0,
                              -std::sin(angle), 0, std::cos(angle) * 0.5f, 0};
    if (!context.CmdBuildTopLevelAccelerationStructure(tlas, instances, ACCELERATION_STRUCTURE_UPDATE))
      return 1;
    if (!context.CmdBindRayTracingProgram(trace))
      return 1;
    if (!context.CmdBindGroup(0, trace_group))
      return 1;
    if (!context.CmdDispatchRays(extent.width, extent.height))
      return 1;
    if (!context.CmdBindProgram(program))
      return 1;
    if (!context.CmdBindGroup(0, display_group))
      return 1;
    if (!context.CmdBeginRendering(image, ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}))
      return 1;
    if (!context.CmdSetViewport({0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1}))
      return 1;
    if (!context.CmdSetScissor({0, 0, extent.width, extent.height}))
      return 1;
    if (!context.CmdDraw(3))
      return 1;
    if (!context.CmdEndRendering())
      return 1;
    if (!headless) {
      if (!context.CmdPresent(window))
        return 1;
    }
    if (!capture.Record(context.Get(), image.Get(), window.Get()))
      return 1;
    SubmissionToken token = queue.Submit(std::move(context));
    capture.Submitted(token.Get());
    if (!token)
      return 1;
    if (headless)
      pending[slot] = std::move(token);
    ++rendered;
    fps.TickFrame();
    auto now = std::chrono::steady_clock::now();
    if (!headless && now - title_time >= std::chrono::milliseconds(250)) {
      if (!window.SetTitle(fmt::format("RHIX raytracing - FPS: {:.1f} - P: toggle VSync", fps.GetFPS())))
        return 1;
      title_time = now;
    }
  }
  if (headless) {
    for (const auto &token : pending) {
      if (token) {
        if (!token.WaitHost())
          return 1;
      }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - benchmark_start).count();
    std::cout << fmt::format(
        "raytracing: headless 1280x720, {} frames, {:.6f} s, {:.2f} FPS (3 in flight, 60 warmup)\n", rendered, seconds,
        rendered / seconds);
  }
  if (!capture.Save())
    return 1;
  return 0;
}
