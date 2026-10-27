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

struct RayQueryModule::State {
  bool initialized{};
  bool headless{};
  std::string capture_directory{};
  grassland::rhix::Device device{};
  grassland::rhix::DownloadToken last_frame{};
  ImageDesc last_frame_desc{};
  ColorSpace last_frame_space{};
  float last_frame_white_scale{};
  PresentMode requested{};
  bool present_mode_changed{};
  Window window{};
  Queue queue{};
  std::array<Image, 3> targets{};
  std::array<SubmissionToken, 3> pending{};
  ComputeProgram trace{};
  Program program{};
  TopLevelAccelerationStructure tlas{};
  std::vector<grassland::rhix::RayTracingInstance> instances{};
  Buffer camera_buffer{};
  BindingGroup trace_group{};
  BindingGroup display_group{};
  Image output{};
  grassland::FPSCounter fps{};
  std::chrono::steady_clock::time_point title_time{};
  std::chrono::steady_clock::time_point benchmark_start{};
  std::chrono::steady_clock::time_point animation_start{};
};

RayQueryModule::RayQueryModule(grassland::rhix::Core core, grassland::rhix::Device device)
    : ModuleBase(std::move(core), std::move(device)) {
}

RayQueryModule::~RayQueryModule() = default;

int RayQueryModule::OnInit(const ModuleOptions &options) {
  state_ = std::make_unique<State>();

  state_->headless = options.headless;
  state_->capture_directory = options.capture_directory;
  using namespace grassland::rhi;
  using namespace grassland::rhix;
  state_->device = GetDevice();
  if (!state_->device.SupportRayQuery()) {
    std::cout << "ray_query: SKIPPED (selected device does not support this feature)\n";
    {
      RequestClose();
      return 0;
    }
  }

  state_->last_frame_desc = ImageDesc{};
  state_->last_frame_space = COLOR_SPACE_SRGB_NONLINEAR;
  state_->last_frame_white_scale = 1.0f;
  state_->requested = PRESENT_MODE_VSYNC;
  state_->present_mode_changed = false;

  if (!state_->headless) {
    state_->window = state_->device.CreateGLFWWindow({1280, 720, "RHIX ray_query - P: toggle VSync"});
    if (!state_->window)
      return 1;
  }
  state_->queue = state_->device.GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!state_->queue)
    return 1;

  if (state_->headless)
    for (auto &target : state_->targets) {
      target = state_->device.CreateImage({1280, 720, IMAGE_FORMAT_R8G8B8A8_UNORM});
      if (!target)
        return 1;
    }
  const auto &embedded = shader_list.at("modules/ray_query/shaders/trace.slang");
  std::string source(embedded.first, embedded.second);
  PipelineLayoutDesc trace_layout{{{{{0, BINDING_ACCELERATION_STRUCTURE, 1, SHADER_VISIBILITY_COMPUTE},
                                     {1, BINDING_RW_TEXTURE2D, 1, SHADER_VISIBILITY_COMPUTE},
                                     {2, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_COMPUTE}}}}};
  Shader cs = state_->device.CreateShaderFromSource(source, "CSMain", SHADER_STAGE_COMPUTE);
  if (!cs)
    return 1;
  state_->trace = state_->device.CreateComputeProgram({cs.Get(), trace_layout});
  if (!state_->trace)
    return 1;

  const auto &display_source = shader_list.at("modules/ray_query/shaders/display.slang");
  std::string display(display_source.first, display_source.second);
  Shader vs = state_->device.CreateShaderFromSource(display, "VSMain", SHADER_STAGE_VERTEX);
  if (!vs)
    return 1;
  Shader ps = state_->device.CreateShaderFromSource(display, "PSMain", SHADER_STAGE_FRAGMENT);
  if (!ps)
    return 1;
  ProgramDesc graphics;
  graphics.vertex_shader = vs.Get();
  graphics.fragment_shader = ps.Get();
  graphics.pipeline_layout.groups = {{{{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT}}}};
  {
    Image first;
    if (state_->headless)
      first = state_->targets[0];
    else
      first = state_->window.GetNextImage();
    if (!first)
      return 1;
    graphics.color_formats = {first.GetDesc().format};
  }
  state_->program = state_->device.CreateProgram(graphics);
  if (!state_->program)
    return 1;
  BottomLevelAccelerationStructureDesc triangles;
  triangles.geometries = {{3, 12, 0}};
  triangles.allow_compaction = true;
  BottomLevelAccelerationStructureDesc boxes;
  boxes.aabbs = {{1, 24, true}};
  BottomLevelAccelerationStructure triangle = state_->device.CreateBottomLevelAccelerationStructure(triangles);
  if (!triangle)
    return 1;
  BottomLevelAccelerationStructure sphere = state_->device.CreateBottomLevelAccelerationStructure(boxes);
  if (!sphere)
    return 1;
  state_->tlas = state_->device.CreateTopLevelAccelerationStructure({2, true});
  if (!state_->tlas)
    return 1;
  {
    const float vertices[] = {-1, -1, 0, 1, -1, 0, 0, 1, 0};
    const float bounds[] = {-1, -1, -1, 1, 1, 1};
    Buffer vertex_buffer = state_->device.CreateBuffer({sizeof(vertices)});
    if (!vertex_buffer)
      return 1;
    Buffer aabb_buffer = state_->device.CreateBuffer({sizeof(bounds)});
    if (!aabb_buffer)
      return 1;
    CommandContext build = state_->device.CreateCommandContext();
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
    SubmissionToken token = state_->queue.Submit(std::move(build));
    if (!token)
      return 1;
    if (!token.WaitHost())
      return 1;
    triangles.compacted_size = triangle.GetCompactedSize();
    if (!triangles.compacted_size)
      return 1;
    BottomLevelAccelerationStructure compact = state_->device.CreateBottomLevelAccelerationStructure(triangles);
    if (!compact)
      return 1;
    CommandContext copy = state_->device.CreateCommandContext();
    if (!copy)
      return 1;
    if (!copy.CmdCompactAccelerationStructure(triangle, compact))
      return 1;
    token = state_->queue.Submit(std::move(copy));
    if (!token)
      return 1;
    if (!token.WaitHost())
      return 1;
    triangle = std::move(compact);
  }
  state_->instances = std::vector<grassland::rhix::RayTracingInstance>(2);
  state_->instances[0].blas = triangle;
  state_->instances[1].blas = sphere;
  state_->instances[1].transform[3] = 2.0f;
  state_->instances[1].hit_group_offset = 0;
  {
    CommandContext build = state_->device.CreateCommandContext();
    if (!build)
      return 1;
    if (!build.CmdBuildTopLevelAccelerationStructure(state_->tlas, state_->instances))
      return 1;
    SubmissionToken token = state_->queue.Submit(std::move(build));
    if (!token)
      return 1;
    if (!token.WaitHost())
      return 1;
  }

  state_->camera_buffer = state_->device.CreateBuffer({256});
  if (!state_->camera_buffer)
    return 1;
  state_->trace_group = state_->device.CreateBindingGroup(trace_layout.groups[0]);
  if (!state_->trace_group)
    return 1;
  if (!state_->trace_group.BindResource(0, state_->tlas))
    return 1;
  state_->display_group = state_->device.CreateBindingGroup(graphics.pipeline_layout.groups[0]);
  if (!state_->display_group)
    return 1;

  if (!state_->trace_group.BindResource(2, state_->camera_buffer))
    return 1;
  if (!state_->headless) {
    state_->window.KeyEvent().RegisterCallback([this](int key, int, int action, int) {
      if (action != GLFW_PRESS)
        return;
      if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(state_->window.GLFWWindow(), GLFW_TRUE);
      if (key == GLFW_KEY_P) {
        state_->requested = state_->requested == PRESENT_MODE_VSYNC ? PRESENT_MODE_CONTINUOUS : PRESENT_MODE_VSYNC;
        state_->present_mode_changed = true;
      }
    });
  }

  state_->title_time = std::chrono::steady_clock::now();
  // Warm up allocations and pipelines before measuring completed GPU work.
  state_->benchmark_start = std::chrono::steady_clock::now();
  state_->animation_start = std::chrono::steady_clock::now();
  SetWarmupFrames(state_->headless ? 60 : 0);

  state_->initialized = true;
  return 0;
}

int RayQueryModule::OnUpdate() {
  if (!state_->headless) {
    if (state_->window.ShouldClose()) {
      RequestClose();
      return 0;
    }
    if (state_->window.IsSwapChainInvalid()) {
      if (!state_->window.RebuildSwapChain())
        return 1;
    }
    if (state_->present_mode_changed) {
      state_->present_mode_changed = false;
      if (!state_->window.SetPresentMode(state_->requested))
        return 1;
    }
  }

  return 0;
}

int RayQueryModule::OnRender() {
  const size_t slot = (static_cast<int64_t>(GetFrameIndex()) + 60) % state_->targets.size();
  if (state_->headless) {
    if (GetFrameIndex() == 0) {
      for (auto &token : state_->pending) {
        if (token) {
          if (!token.WaitHost())
            return 1;
          token = {};
        }
      }
      state_->benchmark_start = std::chrono::steady_clock::now();
    }
    if (state_->pending[slot]) {
      if (!state_->pending[slot].WaitHost())
        return 1;
      state_->pending[slot] = {};
    }
  }
  Image image;
  if (state_->headless)
    image = state_->targets[slot];
  else
    image = state_->window.GetNextImage();
  if (!image) {
    if (state_->headless)
      return 1;
    state_->fps.Reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return 0;
  }
  auto extent = image.GetDesc();
  CommandContext context = state_->device.CreateCommandContext();
  if (!context)
    return 1;
  if (!state_->output || state_->output.GetDesc().width != extent.width ||
      state_->output.GetDesc().height != extent.height) {
    state_->output = state_->device.CreateImage({extent.width, extent.height, IMAGE_FORMAT_R8G8B8A8_UNORM});
    if (!state_->output)
      return 1;
    if (!state_->trace_group.BindResource(1, state_->output))
      return 1;
    if (!state_->display_group.BindResource(0, state_->output))
      return 1;
    CameraInfo camera{
        glm::inverse(glm::perspective(glm::radians(60.0f), float(extent.width) / float(extent.height), 0.1f, 10.0f)),
        glm::inverse(glm::lookAt(glm::vec3{0, 0, 5}, glm::vec3{0, 0, 0}, glm::vec3{0, 1, 0}))};
    if (!context.CmdUploadBuffer(state_->camera_buffer, 0, &camera, sizeof(camera)))
      return 1;
  }
  // Match graphics_hello: one Y-axis revolution every two seconds.
  const float angle =
      state_->headless
          ? float(GetFrameIndex() + 60) * 0.01f
          : std::chrono::duration<float>(std::chrono::steady_clock::now() - state_->animation_start).count() *
                glm::pi<float>();
  state_->instances[0].transform = {std::cos(angle),  0, std::sin(angle), -2, 0, 1, 0, 0,
                                    -std::sin(angle), 0, std::cos(angle), 0};
  state_->instances[1].transform = {std::cos(angle),  0, std::sin(angle) * 0.5f, 2, 0, 1, 0, 0,
                                    -std::sin(angle), 0, std::cos(angle) * 0.5f, 0};
  if (!context.CmdBuildTopLevelAccelerationStructure(state_->tlas, state_->instances, ACCELERATION_STRUCTURE_UPDATE))
    return 1;
  if (!context.CmdBindComputeProgram(state_->trace))
    return 1;
  if (!context.CmdBindGroup(0, state_->trace_group))
    return 1;
  if (!context.CmdDispatch((extent.width + 7) / 8, (extent.height + 7) / 8))
    return 1;
  if (!context.CmdBindProgram(state_->program))
    return 1;
  if (!context.CmdBindGroup(0, state_->display_group))
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
  if (!state_->headless) {
    if (!context.CmdPresent(state_->window))
      return 1;
  }
  if (!state_->capture_directory.empty()) {
    state_->last_frame = context.CmdDownloadImage(image);
    if (!state_->last_frame)
      return 1;
    state_->last_frame_desc = image.GetDesc();
    state_->last_frame_space = state_->window ? state_->window.GetColorSpace() : COLOR_SPACE_SRGB_NONLINEAR;
    state_->last_frame_white_scale = state_->window ? state_->window.GetHDRReferenceWhiteScale() : 1.0f;
  }
  SubmissionToken token = state_->queue.Submit(std::move(context));

  if (!token)
    return 1;
  if (state_->headless)
    state_->pending[slot] = std::move(token);
  CompleteFrame();
  state_->fps.TickFrame();
  auto now = std::chrono::steady_clock::now();
  if (!state_->headless && now - state_->title_time >= std::chrono::milliseconds(250)) {
    if (!state_->window.SetTitle(fmt::format("RHIX ray_query - FPS: {:.1f} - P: toggle VSync", state_->fps.GetFPS())))
      return 1;
    state_->title_time = now;
  }

  return 0;
}

int RayQueryModule::OnTerminate() {
  int result = 0;
  if (state_ && state_->initialized) {
    result = [&]() -> int {
      if (state_->headless) {
        for (const auto &token : state_->pending) {
          if (token) {
            if (!token.WaitHost())
              return 1;
          }
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - state_->benchmark_start).count();
        std::cout << fmt::format(
            "ray_query: headless 1280x720, {} frames, {:.6f} s, {:.2f} FPS (3 in flight, 60 warmup)\n", GetFrameIndex(),
            seconds, GetFrameIndex() / seconds);
      }
      if (!state_->capture_directory.empty()) {
        if (!state_->last_frame || !state_->last_frame.WaitHost())
          return 1;
        std::vector<uint8_t> pixels(state_->last_frame.GetSize());
        if (!state_->last_frame.CopyTo(pixels.data(), pixels.size()) ||
            !SaveImage(state_->capture_directory, "ray_query", pixels, state_->last_frame_desc,
                       state_->last_frame_space, state_->last_frame_white_scale))
          return 1;
      }
      return 0;
    }();
  }
  state_.reset();
  return result;
}
