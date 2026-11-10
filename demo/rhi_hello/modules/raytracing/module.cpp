#include <fmt/format.h>

#include <chrono>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <string>
#include <thread>

#include "grassland/rhi/rhi.h"
#include "grassland/util/fps_counter.h"
#include "snowberg/tui/tui_message_box.h"

// Evaluate once and stop recording immediately when a command fails.
#define RETURN_IF_FAIL(expression) \
  do {                             \
    if (!(expression))             \
      return 1;                    \
  } while (false)

namespace {
#include "shaders/display.slang.h"
#include "shaders/trace.slang.h"
}  // namespace

int RunRayTracing(grassland::rhi::IDevice *device) {
  using namespace grassland::rhi;
  if (!device->SupportRayTracingPipeline()) {
    snowberg::tui::MessageBox message;
    message.Show("raytracing", "The selected device does not support ray tracing pipeline.");
    return 0;
  }
  PresentMode requested = PRESENT_MODE_VSYNC;
  bool present_mode_changed = false;
  ObjPtr<IWindow> window = device->CreateGLFWWindow({1280, 720, "RHI raytracing - P: toggle VSync"});
  if (!window)
    return 1;

  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue)
    return 1;

  std::string source(reinterpret_cast<const char *>(shaders_trace_slang), shaders_trace_slang_len);
  PipelineLayoutDesc trace_layout{{{{{0, BINDING_ACCELERATION_STRUCTURE, 1, SHADER_VISIBILITY_RAY_GENERATION},
                                     {1, BINDING_RW_TEXTURE2D, 1, SHADER_VISIBILITY_RAY_GENERATION},
                                     {2, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_RAY_GENERATION}}}}};
  ObjPtr<IShader> raygen = device->CreateShaderFromSource(source, "RayGenMain", SHADER_STAGE_RAY_GENERATION);
  if (!raygen)
    return 1;
  ObjPtr<IShader> miss = device->CreateShaderFromSource(source, "MissMain", SHADER_STAGE_MISS);
  if (!miss)
    return 1;
  ObjPtr<IShader> hit = device->CreateShaderFromSource(source, "ClosestHitMain", SHADER_STAGE_CLOSEST_HIT);
  if (!hit)
    return 1;
  ObjPtr<IShader> sphere_hit = device->CreateShaderFromSource(source, "SphereClosestHitMain", SHADER_STAGE_CLOSEST_HIT);
  if (!sphere_hit)
    return 1;
  ObjPtr<IShader> intersection =
      device->CreateShaderFromSource(source, "SphereIntersectionMain", SHADER_STAGE_INTERSECTION);
  if (!intersection)
    return 1;
  ObjPtr<IShader> callable = device->CreateShaderFromSource(source, "CallableMain", SHADER_STAGE_CALLABLE);
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
  ObjPtr<IRayTracingProgram> trace = device->CreateRayTracingProgram(desc);
  if (!trace)
    return 1;

  std::string display(reinterpret_cast<const char *>(shaders_display_slang), shaders_display_slang_len);
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
    ObjPtr<IImage> first = window->GetNextImage();
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
    RETURN_IF_FAIL(build->CmdUploadBuffer(vertex_buffer.Get(), 0, vertices, sizeof(vertices)));
    RETURN_IF_FAIL(build->CmdUploadBuffer(aabb_buffer.Get(), 0, bounds, sizeof(bounds)));
    RETURN_IF_FAIL(build->CmdBuildBottomLevelAccelerationStructure(triangle.Get(), {{{vertex_buffer.Get()}, {}}}));
    RETURN_IF_FAIL(build->CmdBuildBottomLevelAccelerationStructure(sphere.Get(), {{{aabb_buffer.Get()}, {}}}));
    RETURN_IF_FAIL(build->CmdQueryCompactedSize(triangle.Get()));
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
    RETURN_IF_FAIL(copy->CmdCompactAccelerationStructure(triangle.Get(), compact.Get()));
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
  instances[1].hit_group_offset = 1;
  {
    ObjPtr<ICommandContext> build = device->CreateCommandContext();
    if (!build)
      return 1;
    RETURN_IF_FAIL(build->CmdBuildTopLevelAccelerationStructure(tlas.Get(), instances));
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

  window->KeyEvent().RegisterCallback([&](int key, int, int action, int) {
    if (action != GLFW_PRESS)
      return;
    if (key == GLFW_KEY_ESCAPE)
      glfwSetWindowShouldClose(window->GLFWWindow(), GLFW_TRUE);
    if (key == GLFW_KEY_P) {
      requested = requested == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
      present_mode_changed = true;
    }
  });

  grassland::FPSCounter fps;
  auto title_time = std::chrono::steady_clock::now();
  const auto animation_start = std::chrono::steady_clock::now();
  while (true) {
    if (!PollEvents())
      return 1;
    if (window->ShouldClose())
      break;
    if (window->IsSwapChainInvalid()) {
      if (!window->RebuildSwapChain())
        return 1;
    }
    if (present_mode_changed) {
      present_mode_changed = false;
      if (!window->SetPresentMode(requested))
        return 1;
    }

    ObjPtr<IImage> image = window->GetNextImage();
    if (!image) {
      fps.Reset();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    auto extent = image->GetDesc();
    ObjPtr<ICommandContext> context = device->CreateCommandContext();
    if (!context)
      return 1;
    if (!output || output->GetDesc().width != extent.width || output->GetDesc().height != extent.height) {
      output = device->CreateImage({extent.width, extent.height, IMAGE_FORMAT_R8G8B8A8_UNORM});
      if (!output)
        return 1;
      if (!trace_group->BindResource(1, output.Get()))
        return 1;
      if (!display_group->BindResource(0, output.Get()))
        return 1;
      CameraInfo camera{
          glm::inverse(glm::perspective(glm::radians(60.0f), float(extent.width) / float(extent.height), 0.1f, 10.0f)),
          glm::inverse(glm::lookAt(glm::vec3{0, 0, 5}, glm::vec3{0, 0, 0}, glm::vec3{0, 1, 0}))};
      RETURN_IF_FAIL(context->CmdUploadBuffer(camera_buffer.Get(), 0, &camera, sizeof(camera)));
    }
    // Match graphics_hello: one Y-axis revolution every two seconds.
    const float angle =
        std::chrono::duration<float>(std::chrono::steady_clock::now() - animation_start).count() * glm::pi<float>();
    instances[0].transform = {std::cos(angle),  0, std::sin(angle), -2, 0, 1, 0, 0,
                              -std::sin(angle), 0, std::cos(angle), 0};
    instances[1].transform = {std::cos(angle),  0, std::sin(angle) * 0.5f, 2, 0, 1, 0, 0,
                              -std::sin(angle), 0, std::cos(angle) * 0.5f, 0};
    RETURN_IF_FAIL(
        context->CmdBuildTopLevelAccelerationStructure(tlas.Get(), instances, ACCELERATION_STRUCTURE_UPDATE));
    RETURN_IF_FAIL(context->CmdBindRayTracingProgram(trace.Get()));
    RETURN_IF_FAIL(context->CmdBindGroup(0, trace_group.Get()));
    RETURN_IF_FAIL(context->CmdDispatchRays(extent.width, extent.height));
    RETURN_IF_FAIL(context->CmdBindProgram(program.Get()));
    RETURN_IF_FAIL(context->CmdBindGroup(0, display_group.Get()));
    RETURN_IF_FAIL(context->CmdBeginRendering(
        {{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}));
    RETURN_IF_FAIL(
        context->CmdSetViewport({0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1}));
    RETURN_IF_FAIL(context->CmdSetScissor({0, 0, extent.width, extent.height}));
    RETURN_IF_FAIL(context->CmdDraw(3));
    RETURN_IF_FAIL(context->CmdEndRendering());

    RETURN_IF_FAIL(context->CmdPresent(window.Get()));

    ObjPtr<ISubmissionToken> token = queue->Submit(context.Get());

    if (!token)
      return 1;

    fps.TickFrame();
    auto now = std::chrono::steady_clock::now();
    if (now - title_time >= std::chrono::milliseconds(250)) {
      if (!window->SetTitle(fmt::format("RHI raytracing - FPS: {:.1f} - P: toggle VSync", fps.GetFPS())))
        return 1;
      title_time = now;
    }
  }

  return 0;
}

#undef RETURN_IF_FAIL
