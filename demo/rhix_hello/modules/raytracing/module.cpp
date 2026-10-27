#include "module.h"

#include <fmt/format.h>

#include <array>
#include <chrono>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <map>
#include <thread>

#include "../../save_image.h"
#include "grassland/rhi/rhix/rhix.h"
#include "grassland/util/fps_counter.h"
#include "grassland/util/virtual_file_system.h"

namespace {
struct CameraInfo {
  glm::mat4 screen_to_camera;
  glm::mat4 camera_to_world;
};

#include "built_in_shaders.inl"
}  // namespace

using namespace grassland::rhi;
using namespace grassland::rhix;

int RayTracingModule::OnInit(const ModuleOptions &options) {
  headless_ = options.headless;
  capture_directory_ = options.capture_directory;
  using namespace grassland::rhi;
  using namespace grassland::rhix;
  device_ = GetDevice();
  if (!device_.SupportRayTracingPipeline()) {
    std::cout << "raytracing: SKIPPED (selected device does not support this feature)\n";
    {
      RequestClose();
      return 0;
    }
  }

  last_frame_desc_ = ImageDesc{};
  last_frame_space_ = COLOR_SPACE_SRGB_NONLINEAR;
  last_frame_white_scale_ = 1.0f;
  requested_ = PRESENT_MODE_VSYNC;
  present_mode_changed_ = false;

  if (!headless_) {
    window_ = device_.CreateGLFWWindow({1280, 720, "RHIX raytracing - P: toggle VSync"});
    if (!window_)
      return 1;
  }
  queue_ = device_.GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue_)
    return 1;

  if (headless_)
    for (auto &target : targets_) {
      target = device_.CreateImage({1280, 720, IMAGE_FORMAT_R8G8B8A8_UNORM});
      if (!target)
        return 1;
    }
  const auto &embedded = shader_list.at("modules/raytracing/shaders/trace.slang");
  std::string source(embedded.first, embedded.second);
  PipelineLayoutDesc trace_layout{{{{{0, BINDING_ACCELERATION_STRUCTURE, 1, SHADER_VISIBILITY_RAY_GENERATION},
                                     {1, BINDING_RW_TEXTURE2D, 1, SHADER_VISIBILITY_RAY_GENERATION},
                                     {2, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_RAY_GENERATION}}}}};
  Shader raygen = device_.CreateShaderFromSource(source, "RayGenMain", SHADER_STAGE_RAY_GENERATION);
  if (!raygen)
    return 1;
  Shader miss = device_.CreateShaderFromSource(source, "MissMain", SHADER_STAGE_MISS);
  if (!miss)
    return 1;
  Shader hit = device_.CreateShaderFromSource(source, "ClosestHitMain", SHADER_STAGE_CLOSEST_HIT);
  if (!hit)
    return 1;
  Shader sphere_hit = device_.CreateShaderFromSource(source, "SphereClosestHitMain", SHADER_STAGE_CLOSEST_HIT);
  if (!sphere_hit)
    return 1;
  Shader intersection = device_.CreateShaderFromSource(source, "SphereIntersectionMain", SHADER_STAGE_INTERSECTION);
  if (!intersection)
    return 1;
  Shader callable = device_.CreateShaderFromSource(source, "CallableMain", SHADER_STAGE_CALLABLE);
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
  trace_ = device_.CreateRayTracingProgram(desc);
  if (!trace_)
    return 1;

  const auto &display_source = shader_list.at("modules/raytracing/shaders/display.slang");
  std::string display(display_source.first, display_source.second);
  Shader vs = device_.CreateShaderFromSource(display, "VSMain", SHADER_STAGE_VERTEX);
  if (!vs)
    return 1;
  Shader ps = device_.CreateShaderFromSource(display, "PSMain", SHADER_STAGE_FRAGMENT);
  if (!ps)
    return 1;
  ProgramDesc graphics;
  graphics.vertex_shader = vs.Get();
  graphics.fragment_shader = ps.Get();
  graphics.pipeline_layout.groups = {{{{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT}}}};
  {
    Image first;
    if (headless_)
      first = targets_[0];
    else
      first = window_.GetNextImage();
    if (!first)
      return 1;
    graphics.color_formats = {first.GetDesc().format};
  }
  program_ = device_.CreateProgram(graphics);
  if (!program_)
    return 1;
  BottomLevelAccelerationStructureDesc triangles;
  triangles.geometries = {{3, 12, 0}};
  triangles.allow_compaction = true;
  BottomLevelAccelerationStructureDesc boxes;
  boxes.aabbs = {{1, 24, true}};
  BottomLevelAccelerationStructure triangle = device_.CreateBottomLevelAccelerationStructure(triangles);
  if (!triangle)
    return 1;
  BottomLevelAccelerationStructure sphere = device_.CreateBottomLevelAccelerationStructure(boxes);
  if (!sphere)
    return 1;
  tlas_ = device_.CreateTopLevelAccelerationStructure({2, true});
  if (!tlas_)
    return 1;
  {
    const float vertices[] = {-1, -1, 0, 1, -1, 0, 0, 1, 0};
    const float bounds[] = {-1, -1, -1, 1, 1, 1};
    Buffer vertex_buffer = device_.CreateBuffer({sizeof(vertices)});
    if (!vertex_buffer)
      return 1;
    Buffer aabb_buffer = device_.CreateBuffer({sizeof(bounds)});
    if (!aabb_buffer)
      return 1;
    CommandContext build = device_.CreateCommandContext();
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
    SubmissionToken token = queue_.Submit(std::move(build));
    if (!token)
      return 1;
    if (!token.WaitHost())
      return 1;
    triangles.compacted_size = triangle.GetCompactedSize();
    if (!triangles.compacted_size)
      return 1;
    BottomLevelAccelerationStructure compact = device_.CreateBottomLevelAccelerationStructure(triangles);
    if (!compact)
      return 1;
    CommandContext copy = device_.CreateCommandContext();
    if (!copy)
      return 1;
    if (!copy.CmdCompactAccelerationStructure(triangle, compact))
      return 1;
    token = queue_.Submit(std::move(copy));
    if (!token)
      return 1;
    if (!token.WaitHost())
      return 1;
    triangle = std::move(compact);
  }
  instances_ = std::vector<grassland::rhix::RayTracingInstance>(2);
  instances_[0].blas = triangle;
  instances_[1].blas = sphere;
  instances_[1].transform[3] = 2.0f;
  instances_[1].hit_group_offset = 1;
  {
    CommandContext build = device_.CreateCommandContext();
    if (!build)
      return 1;
    if (!build.CmdBuildTopLevelAccelerationStructure(tlas_, instances_))
      return 1;
    SubmissionToken token = queue_.Submit(std::move(build));
    if (!token)
      return 1;
    if (!token.WaitHost())
      return 1;
  }

  camera_buffer_ = device_.CreateBuffer({256});
  if (!camera_buffer_)
    return 1;
  trace_group_ = device_.CreateBindingGroup(trace_layout.groups[0]);
  if (!trace_group_)
    return 1;
  if (!trace_group_.BindResource(0, tlas_))
    return 1;
  display_group_ = device_.CreateBindingGroup(graphics.pipeline_layout.groups[0]);
  if (!display_group_)
    return 1;

  if (!trace_group_.BindResource(2, camera_buffer_))
    return 1;
  if (!headless_) {
    window_.KeyEvent().RegisterCallback([this](int key, int, int action, int) {
      if (action != GLFW_PRESS)
        return;
      if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window_.GLFWWindow(), GLFW_TRUE);
      if (key == GLFW_KEY_P) {
        requested_ = requested_ == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
        present_mode_changed_ = true;
      }
    });
  }

  title_time_ = std::chrono::steady_clock::now();
  // Warm up allocations and pipelines before measuring completed GPU work.
  benchmark_start_ = std::chrono::steady_clock::now();
  animation_start_ = std::chrono::steady_clock::now();
  SetWarmupFrames(headless_ ? 60 : 0);

  initialized_ = true;
  return 0;
}

int RayTracingModule::OnUpdate() {
  if (!headless_) {
    if (window_.ShouldClose()) {
      RequestClose();
      return 0;
    }
    if (window_.IsSwapChainInvalid()) {
      if (!window_.RebuildSwapChain())
        return 1;
    }
    if (present_mode_changed_) {
      present_mode_changed_ = false;
      if (!window_.SetPresentMode(requested_))
        return 1;
    }
  }

  return 0;
}

int RayTracingModule::OnRender() {
  const size_t slot = (static_cast<int64_t>(GetFrameIndex()) + 60) % targets_.size();
  if (headless_) {
    if (GetFrameIndex() == 0) {
      for (auto &token : pending_) {
        if (token) {
          if (!token.WaitHost())
            return 1;
          token = {};
        }
      }
      benchmark_start_ = std::chrono::steady_clock::now();
    }
    if (pending_[slot]) {
      if (!pending_[slot].WaitHost())
        return 1;
      pending_[slot] = {};
    }
  }
  Image image;
  if (headless_)
    image = targets_[slot];
  else
    image = window_.GetNextImage();
  if (!image) {
    if (headless_)
      return 1;
    fps_.Reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return 0;
  }
  auto extent = image.GetDesc();
  CommandContext context = device_.CreateCommandContext();
  if (!context)
    return 1;
  if (!output_ || output_.GetDesc().width != extent.width || output_.GetDesc().height != extent.height) {
    output_ = device_.CreateImage({extent.width, extent.height, IMAGE_FORMAT_R8G8B8A8_UNORM});
    if (!output_)
      return 1;
    if (!trace_group_.BindResource(1, output_))
      return 1;
    if (!display_group_.BindResource(0, output_))
      return 1;
    CameraInfo camera{
        glm::inverse(glm::perspective(glm::radians(60.0f), float(extent.width) / float(extent.height), 0.1f, 10.0f)),
        glm::inverse(glm::lookAt(glm::vec3{0, 0, 5}, glm::vec3{0, 0, 0}, glm::vec3{0, 1, 0}))};
    if (!context.CmdUploadBuffer(camera_buffer_, 0, &camera, sizeof(camera)))
      return 1;
  }
  // Match graphics_hello: one Y-axis revolution every two seconds.
  const float angle = headless_
                          ? float(GetFrameIndex() + 60) * 0.01f
                          : std::chrono::duration<float>(std::chrono::steady_clock::now() - animation_start_).count() *
                                glm::pi<float>();
  instances_[0].transform = {std::cos(angle),  0, std::sin(angle), -2, 0, 1, 0, 0,
                             -std::sin(angle), 0, std::cos(angle), 0};
  instances_[1].transform = {std::cos(angle),  0, std::sin(angle) * 0.5f, 2, 0, 1, 0, 0,
                             -std::sin(angle), 0, std::cos(angle) * 0.5f, 0};
  if (!context.CmdBuildTopLevelAccelerationStructure(tlas_, instances_, ACCELERATION_STRUCTURE_UPDATE))
    return 1;
  if (!context.CmdBindRayTracingProgram(trace_))
    return 1;
  if (!context.CmdBindGroup(0, trace_group_))
    return 1;
  if (!context.CmdDispatchRays(extent.width, extent.height))
    return 1;
  if (!context.CmdBindProgram(program_))
    return 1;
  if (!context.CmdBindGroup(0, display_group_))
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
  if (!headless_) {
    if (!context.CmdPresent(window_))
      return 1;
  }
  if (!capture_directory_.empty()) {
    last_frame_ = context.CmdDownloadImage(image);
    if (!last_frame_)
      return 1;
    last_frame_desc_ = image.GetDesc();
    last_frame_space_ = window_ ? window_.GetColorSpace() : COLOR_SPACE_SRGB_NONLINEAR;
    last_frame_white_scale_ = window_ ? window_.GetHDRReferenceWhiteScale() : 1.0f;
  }
  SubmissionToken token = queue_.Submit(std::move(context));

  if (!token)
    return 1;
  if (headless_)
    pending_[slot] = std::move(token);
  CompleteFrame();
  fps_.TickFrame();
  auto now = std::chrono::steady_clock::now();
  if (!headless_ && now - title_time_ >= std::chrono::milliseconds(250)) {
    if (!window_.SetTitle(fmt::format("RHIX raytracing - FPS: {:.1f} - P: toggle VSync", fps_.GetFPS())))
      return 1;
    title_time_ = now;
  }

  return 0;
}

int RayTracingModule::OnTerminate() {
  int result = 0;
  if (initialized_) {
    result = [&]() -> int {
      if (headless_) {
        for (const auto &token : pending_) {
          if (token) {
            if (!token.WaitHost())
              return 1;
          }
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - benchmark_start_).count();
        std::cout << fmt::format(
            "raytracing: headless 1280x720, {} frames, {:.6f} s, {:.2f} FPS (3 in flight, 60 warmup)\n",
            GetFrameIndex(), seconds, GetFrameIndex() / seconds);
      }
      if (!capture_directory_.empty()) {
        if (!last_frame_ || !last_frame_.WaitHost())
          return 1;
        std::vector<uint8_t> pixels(last_frame_.GetSize());
        if (!last_frame_.CopyTo(pixels.data(), pixels.size()) ||
            !SaveImage(capture_directory_, "raytracing", pixels, last_frame_desc_, last_frame_space_,
                       last_frame_white_scale_))
          return 1;
      }
      return 0;
    }();
  }
  animation_start_ = {};
  benchmark_start_ = {};
  title_time_ = {};
  fps_ = {};
  output_ = {};
  display_group_ = {};
  trace_group_ = {};
  camera_buffer_ = {};
  instances_ = {};
  tlas_ = {};
  program_ = {};
  trace_ = {};
  pending_ = {};
  targets_ = {};
  queue_ = {};
  window_ = {};
  present_mode_changed_ = {};
  requested_ = {};
  last_frame_white_scale_ = {};
  last_frame_space_ = {};
  last_frame_desc_ = {};
  last_frame_ = {};
  device_ = {};
  capture_directory_ = {};
  headless_ = {};
  initialized_ = {};
  return result;
}
