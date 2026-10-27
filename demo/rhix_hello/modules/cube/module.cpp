#include "module.h"

#include <fmt/format.h>

#include <array>
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <utility>

#include "../../save_image.h"
#include "grassland/rhi/rhix/rhix.h"
#include "grassland/util/fps_counter.h"
#include "grassland/util/virtual_file_system.h"

namespace {
#include "built_in_shaders.inl"

struct Vertex {
  float position[3];
  float color[3];
};
}  // namespace

using namespace grassland::rhi;
using namespace grassland::rhix;

struct CubeModule::State {
  bool initialized{};
  bool headless{};
  std::string capture_directory{};
  grassland::rhix::Device device{};
  grassland::rhix::DownloadToken last_frame{};
  grassland::rhix::DownloadToken last_depth{};
  ImageDesc last_depth_desc{};
  ImageDesc last_frame_desc{};
  ColorSpace last_frame_space{};
  float last_frame_white_scale{};
  PresentMode requested{};
  bool present_mode_changed{};
  grassland::rhix::Window window{};
  grassland::rhix::Queue queue{};
  std::array<grassland::rhix::Image, 3> targets{};
  std::array<grassland::rhix::SubmissionToken, 3> pending{};
  grassland::rhix::Program program{};
  grassland::rhix::Buffer vertex_buffer{};
  grassland::rhix::Buffer index_buffer{};
  grassland::rhix::Buffer uniform{};
  grassland::rhix::BindingGroup group{};
  grassland::rhix::Image depth{};
  grassland::FPSCounter fps{};
  std::chrono::steady_clock::time_point title_time{};
  std::chrono::steady_clock::time_point benchmark_start{};
};

CubeModule::CubeModule(grassland::rhix::Core core, grassland::rhix::Device device)
    : ModuleBase(std::move(core), std::move(device)) {
}

CubeModule::~CubeModule() = default;

int CubeModule::OnInit(const ModuleOptions &options) {
  state_ = std::make_unique<State>();

  state_->headless = options.headless;
  state_->capture_directory = options.capture_directory;
  using namespace grassland::rhi;
  using namespace grassland::rhix;
  state_->device = GetDevice();

  state_->last_depth_desc = ImageDesc{};
  state_->last_frame_desc = ImageDesc{};
  state_->last_frame_space = COLOR_SPACE_SRGB_NONLINEAR;
  state_->last_frame_white_scale = 1.0f;
  state_->requested = PRESENT_MODE_VSYNC;
  state_->present_mode_changed = false;

  if (!state_->headless) {
    state_->window = state_->device.CreateGLFWWindow({1280, 720, "RHIX cube - P: toggle VSync"});
    if (!state_->window)
      return 1;
  }
  state_->queue = state_->device.GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!state_->queue)
    return 1;

  if (state_->headless) {
    for (auto &target : state_->targets) {
      target = state_->device.CreateImage({1280, 720, IMAGE_FORMAT_R8G8B8A8_UNORM});
      if (!target)
        return 1;
    }
  }
  const auto &embedded = shader_list.at("modules/cube/shaders/cube.slang");
  std::string source(embedded.first, embedded.second);
  grassland::rhix::Shader vertex_shader = state_->device.CreateShaderFromSource(source, "VSMain", SHADER_STAGE_VERTEX);
  if (!vertex_shader)
    return 1;
  grassland::rhix::Shader fragment_shader =
      state_->device.CreateShaderFromSource(source, "PSMain", SHADER_STAGE_FRAGMENT);
  if (!fragment_shader)
    return 1;
  ProgramDesc desc;
  desc.vertex_shader = vertex_shader.Get();
  desc.fragment_shader = fragment_shader.Get();
  desc.vertex_bindings = {{sizeof(Vertex), false}};
  desc.vertex_attributes = {{0, 0, INPUT_TYPE_FLOAT3, 0}, {1, 0, INPUT_TYPE_FLOAT3, 12}};
  {
    grassland::rhix::Image image;
    if (state_->headless)
      image = state_->targets[0];
    else
      image = state_->window.GetNextImage();
    if (!image)
      return 1;
    desc.color_formats = {image.GetDesc().format};
  }
  desc.pipeline_layout.groups = {{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_VERTEX}}}};
  desc.depth_stencil.format = IMAGE_FORMAT_D32_SFLOAT;
  desc.depth_stencil.depth_test = true;
  desc.depth_stencil.depth_write = true;
  state_->program = state_->device.CreateProgram(desc);
  if (!state_->program)
    return 1;
  const Vertex vertices[] = {{{-1, -1, 1}, {0, 0, 0}}, {{1, -1, 1}, {1, 0, 0}},   {{-1, 1, 1}, {0, 1, 0}},
                             {{1, 1, 1}, {1, 1, 0}},   {{-1, -1, -1}, {0, 0, 1}}, {{1, -1, -1}, {1, 0, 1}},
                             {{-1, 1, -1}, {0, 1, 1}}, {{1, 1, -1}, {1, 1, 1}}};
  const uint32_t indices[] = {0, 1, 2, 2, 1, 3, 2, 3, 6, 6, 3, 7, 6, 7, 4, 4, 7, 5,
                              4, 5, 0, 0, 5, 1, 1, 5, 3, 3, 5, 7, 0, 2, 4, 4, 2, 6};
  state_->vertex_buffer = state_->device.CreateBuffer({sizeof(vertices)});
  if (!state_->vertex_buffer)
    return 1;
  state_->index_buffer = state_->device.CreateBuffer({sizeof(indices)});
  if (!state_->index_buffer)
    return 1;
  state_->uniform = state_->device.CreateBuffer({256});
  if (!state_->uniform)
    return 1;
  state_->group = state_->device.CreateBindingGroup(desc.pipeline_layout.groups[0]);
  if (!state_->group)
    return 1;
  if (!state_->group.BindResource(0, state_->uniform))
    return 1;

  // Upload owns a snapshot and staging leases; later graphics work is ordered on the same native queue.
  {
    grassland::rhix::CommandContext upload = state_->device.CreateCommandContext();
    if (!upload)
      return 1;
    if (!upload.CmdUploadBuffer(state_->vertex_buffer, 0, vertices, sizeof(vertices)))
      return 1;
    if (!upload.CmdUploadBuffer(state_->index_buffer, 0, indices, sizeof(indices)))
      return 1;
    grassland::rhix::SubmissionToken uploaded = state_->queue.Submit(std::move(upload));
    if (!uploaded)
      return 1;
    if (state_->headless) {
      if (!uploaded.WaitHost())
        return 1;
    }
  }
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
  SetWarmupFrames(state_->headless ? 60 : 0);

  state_->initialized = true;
  return 0;
}

int CubeModule::OnUpdate() {
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

int CubeModule::OnRender() {
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
  grassland::rhix::Image image;
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
  grassland::rhix::CommandContext context = state_->device.CreateCommandContext();
  if (!context)
    return 1;
  // GPU copies and draws are ordered on the same queue; the upload takes a CPU snapshot.
  const float angle =
      state_->headless
          ? float(GetFrameIndex() + 60) * .01f
          : float(std::chrono::duration<double>(std::chrono::steady_clock::now() - state_->benchmark_start).count());

  struct Uniform {
    glm::mat4 model, view, projection;
  } ubo;

  ubo.model = glm::rotate(glm::mat4(1), angle, glm::vec3(0, 1, 0));
  ubo.view = glm::lookAt(glm::vec3(0, 0, 5), glm::vec3(0), glm::vec3(0, 1, 0));
  ubo.projection = glm::perspectiveZO(glm::radians(45.0f), float(extent.width) / extent.height, 3.5f, 6.5f);
  if (!context.CmdUploadBuffer(state_->uniform, 0, &ubo, sizeof(ubo)))
    return 1;
  if (!state_->depth || state_->depth.GetDesc().width != extent.width ||
      state_->depth.GetDesc().height != extent.height) {
    state_->depth = state_->device.CreateImage({extent.width, extent.height, IMAGE_FORMAT_D32_SFLOAT});
    if (!state_->depth)
      return 1;
  }
  if (!context.CmdBindProgram(state_->program))
    return 1;
  if (!context.CmdBindGroup(0, state_->group))
    return 1;
  grassland::rhix::RenderingInfo render;
  render.colors = {{image, ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {.6f, .7f, .8f, 1}}};
  render.depth_stencil = state_->depth;
  render.depth_stencil_ops.depth_load_op = ATTACHMENT_LOAD_OP_CLEAR;
  if (!context.CmdBeginRendering(render))
    return 1;
  if (!context.CmdBindVertexBuffer(0, state_->vertex_buffer))
    return 1;
  if (!context.CmdBindIndexBuffer(state_->index_buffer))
    return 1;
  if (!context.CmdSetViewport({0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1}))
    return 1;
  if (!context.CmdSetScissor({0, 0, extent.width, extent.height}))
    return 1;
  if (!context.CmdDrawIndexed(36))
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
    state_->last_depth = context.CmdDownloadImage(state_->depth);
    if (!state_->last_depth)
      return 1;
    state_->last_depth_desc = state_->depth.GetDesc();
    state_->last_frame_desc = image.GetDesc();
    state_->last_frame_space = state_->window ? state_->window.GetColorSpace() : COLOR_SPACE_SRGB_NONLINEAR;
    state_->last_frame_white_scale = state_->window ? state_->window.GetHDRReferenceWhiteScale() : 1.0f;
  }
  grassland::rhix::SubmissionToken token = state_->queue.Submit(std::move(context));

  if (!token)
    return 1;
  if (state_->headless)
    state_->pending[slot] = std::move(token);
  CompleteFrame();
  state_->fps.TickFrame();
  auto now = std::chrono::steady_clock::now();
  if (!state_->headless && now - state_->title_time >= std::chrono::milliseconds(250)) {
    if (!state_->window.SetTitle(fmt::format("RHIX cube - FPS: {:.1f} - P: toggle VSync", state_->fps.GetFPS())))
      return 1;
    state_->title_time = now;
  }

  return 0;
}

int CubeModule::OnTerminate() {
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
        std::cout << fmt::format("cube: headless 1280x720, {} frames, {:.6f} s, {:.2f} FPS (3 in flight, 60 warmup)\n",
                                 GetFrameIndex(), seconds, GetFrameIndex() / seconds);
      }
      if (!state_->capture_directory.empty()) {
        if (!state_->last_frame || !state_->last_frame.WaitHost())
          return 1;
        std::vector<uint8_t> pixels(state_->last_frame.GetSize());
        if (!state_->last_frame.CopyTo(pixels.data(), pixels.size()) ||
            !SaveImage(state_->capture_directory, "cube", pixels, state_->last_frame_desc, state_->last_frame_space,
                       state_->last_frame_white_scale))
          return 1;
        if (!state_->last_depth || !state_->last_depth.WaitHost())
          return 1;
        pixels.resize(state_->last_depth.GetSize());
        if (!state_->last_depth.CopyTo(pixels.data(), pixels.size()) ||
            !SaveImage(state_->capture_directory, "cube_depth", pixels, state_->last_depth_desc,
                       COLOR_SPACE_SRGB_NONLINEAR, 1.0f))
          return 1;
      }
      return 0;
    }();
  }
  state_.reset();
  return result;
}
