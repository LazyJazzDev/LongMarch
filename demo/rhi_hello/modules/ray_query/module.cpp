#include "../../module.h"

#include <fmt/format.h>

#include <array>
#include <chrono>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <map>
#include <thread>

#include "grassland/rhi/rhi.h"
#include "grassland/util/fps_counter.h"
#include "grassland/util/virtual_file_system.h"

namespace {
#include "built_in_shaders.inl"
}

int RunRayQuery(grassland::rhi::BackendAPI api, int frames, bool headless) {
  using namespace grassland::rhi;
  ObjPtr<ICore> core = CreateCore(api);
  if (!core)
    return 1;
  DeviceFeatureRequirement required;
  required.ray_query = true;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection(required);
  if (!device) {
    std::cout << "ray_query: SKIPPED (ray tracing unavailable)\n";
    return 0;
  }
  ObjPtr<IWindow> window;
  if (!headless) {
    window = device->CreateGLFWWindow({1280, 720, "RHI ray_query - P: toggle VSync"});
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
  const auto &embedded = shader_list.at("modules/ray_query/shaders/trace.slang");
  std::string source(embedded.first, embedded.second);
  PipelineLayoutDesc trace_layout{{{{{0, BINDING_ACCELERATION_STRUCTURE, 1, SHADER_VISIBILITY_COMPUTE},
                                     {1, BINDING_RW_TEXTURE2D, 1, SHADER_VISIBILITY_COMPUTE},
                                     {2, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_COMPUTE}}}}};
  ObjPtr<IShader> cs = device->CreateShader({source, "CSMain", SHADER_STAGE_COMPUTE});
  if (!cs)
    return 1;
  ObjPtr<IComputeProgram> trace = device->CreateComputeProgram({cs.Get(), trace_layout});
  if (!trace)
    return 1;

  const auto &display_source = shader_list.at("modules/ray_query/shaders/display.slang");
  std::string display(display_source.first, display_source.second);
  ObjPtr<IShader> vs = device->CreateShader({display, "VSMain", SHADER_STAGE_VERTEX});
  if (!vs)
    return 1;
  ObjPtr<IShader> ps = device->CreateShader({display, "PSMain", SHADER_STAGE_FRAGMENT});
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
  BottomLevelAccelerationStructureDesc triangles;
  triangles.geometries = {{3, 12, 0}};
  triangles.allow_compaction = true;
  BottomLevelAccelerationStructureDesc boxes;
  boxes.aabbs = {{1, 24, true}};
  ObjPtr<IBottomLevelAccelerationStructure> triangle = device->CreateBottomLevelAccelerationStructure(triangles);
  if (!triangle)
    return 1;
  ObjPtr<IBottomLevelAccelerationStructure> sphere = device->CreateBottomLevelAccelerationStructure(boxes);
  if (!sphere)
    return 1;
  ObjPtr<ITopLevelAccelerationStructure> tlas = device->CreateTopLevelAccelerationStructure({2, true});
  if (!tlas)
    return 1;
  {
    const float vertices[] = {-1, -1, 0, 1, -1, 0, 0, 1, 0};
    const float bounds[] = {-1, -1, -1, 1, 1, 1};
    ObjPtr<IBuffer> vertex_buffer = device->CreateBuffer({sizeof(vertices)});
    if (!vertex_buffer)
      return 1;
    ObjPtr<IBuffer> aabb_buffer = device->CreateBuffer({sizeof(bounds)});
    if (!aabb_buffer)
      return 1;
    ObjPtr<ICommandContext> build = device->CreateCommandContext();
    if (!build)
      return 1;
    if (!build->CmdUploadBuffer(vertex_buffer.Get(), 0, vertices, sizeof(vertices)))
      return 1;
    if (!build->CmdUploadBuffer(aabb_buffer.Get(), 0, bounds, sizeof(bounds)))
      return 1;
    if (!build->CmdBuildBottomLevelAccelerationStructure(triangle.Get(), {{{vertex_buffer.Get()}, {}}}))
      return 1;
    if (!build->CmdBuildBottomLevelAccelerationStructure(sphere.Get(), {{{aabb_buffer.Get()}, {}}}))
      return 1;
    if (!build->CmdQueryCompactedSize(triangle.Get()))
      return 1;
    ObjPtr<ISubmissionToken> token = queue->Submit(build.Get());
    if (!token)
      return 1;
    if (!token->WaitHost())
      return 1;
    triangles.compacted_size = triangle->GetCompactedSize();
    if (!triangles.compacted_size)
      return 1;
    ObjPtr<IBottomLevelAccelerationStructure> compact = device->CreateBottomLevelAccelerationStructure(triangles);
    if (!compact)
      return 1;
    ObjPtr<ICommandContext> copy = device->CreateCommandContext();
    if (!copy)
      return 1;
    if (!copy->CmdCompactAccelerationStructure(triangle.Get(), compact.Get()))
      return 1;
    token = queue->Submit(copy.Get());
    if (!token)
      return 1;
    if (!token->WaitHost())
      return 1;
    triangle = std::move(compact);
  }
  std::vector<RayTracingInstance> instances(2);
  instances[0].blas = triangle.Get();
  instances[1].blas = sphere.Get();
  instances[1].transform[3] = 2.0f;
  instances[1].hit_group_offset = 0;
  {
    ObjPtr<ICommandContext> build = device->CreateCommandContext();
    if (!build)
      return 1;
    if (!build->CmdBuildTopLevelAccelerationStructure(tlas.Get(), instances))
      return 1;
    ObjPtr<ISubmissionToken> token = queue->Submit(build.Get());
    if (!token)
      return 1;
    if (!token->WaitHost())
      return 1;
  }

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
  bool previous_p = false;
  PresentMode requested = PRESENT_MODE_VSYNC;
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
    if (!output || output->GetDesc().width != extent.width || output->GetDesc().height != extent.height) {
      output = device->CreateImage({extent.width, extent.height, IMAGE_FORMAT_RGBA8_UNORM});
      if (!output)
        return 1;
      if (!trace_group->BindResource(1, output.Get()))
        return 1;
      if (!display_group->BindResource(0, output.Get()))
        return 1;
      CameraInfo camera{
          glm::inverse(glm::perspective(glm::radians(60.0f), float(extent.width) / float(extent.height), 0.1f, 10.0f)),
          glm::inverse(glm::lookAt(glm::vec3{0, 0, 5}, glm::vec3{0, 0, 0}, glm::vec3{0, 1, 0}))};
      if (!context->CmdUploadBuffer(camera_buffer.Get(), 0, &camera, sizeof(camera)))
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
    if (!context->CmdBuildTopLevelAccelerationStructure(tlas.Get(), instances, ACCELERATION_STRUCTURE_UPDATE))
      return 1;
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
      if (!window->SetTitle(fmt::format("RHI ray_query - FPS: {:.1f} - P: toggle VSync", fps.GetFPS())))
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
    std::cout << fmt::format("ray_query: headless 1280x720, {} frames, {:.6f} s, {:.2f} FPS (3 in flight, 60 warmup)\n",
                             rendered, seconds, rendered / seconds);
  }
  return 0;
}
