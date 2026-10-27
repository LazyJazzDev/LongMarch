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
  float uv[2];
};
}  // namespace

using namespace grassland::rhi;
using namespace grassland::rhix;

int TextureModule::OnInit(const ModuleOptions &options) {
  headless_ = options.headless;
  capture_directory_ = options.capture_directory;
  using namespace grassland::rhi;
  using namespace grassland::rhix;
  device_ = GetDevice();

  last_frame_desc_ = ImageDesc{};
  last_frame_space_ = COLOR_SPACE_SRGB_NONLINEAR;
  last_frame_white_scale_ = 1.0f;
  requested_ = PRESENT_MODE_VSYNC;
  present_mode_changed_ = false;

  if (!headless_) {
    window_ = device_.CreateGLFWWindow({1280, 720, "RHIX texture - P: toggle VSync"});
    if (!window_)
      return 1;
  }
  queue_ = device_.GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue_)
    return 1;

  if (headless_) {
    for (auto &target : targets_) {
      target = device_.CreateImage({1280, 720, IMAGE_FORMAT_R8G8B8A8_UNORM});
      if (!target)
        return 1;
    }
  }
  const auto &embedded = shader_list.at("modules/texture/shaders/texture.slang");
  std::string source(embedded.first, embedded.second);
  grassland::rhix::Shader vertex_shader = device_.CreateShaderFromSource(source, "VSMain", SHADER_STAGE_VERTEX);
  if (!vertex_shader)
    return 1;
  grassland::rhix::Shader fragment_shader = device_.CreateShaderFromSource(source, "PSMain", SHADER_STAGE_FRAGMENT);
  if (!fragment_shader)
    return 1;
  ProgramDesc desc;
  desc.vertex_shader = vertex_shader.Get();
  desc.fragment_shader = fragment_shader.Get();
  desc.vertex_bindings = {{sizeof(Vertex), false}};
  desc.vertex_attributes = {{0, 0, INPUT_TYPE_FLOAT3, 0}, {1, 0, INPUT_TYPE_FLOAT2, 12}};
  {
    grassland::rhix::Image image;
    if (headless_)
      image = targets_[0];
    else
      image = window_.GetNextImage();
    if (!image)
      return 1;
    desc.color_formats = {image.GetDesc().format};
  }
  desc.pipeline_layout.groups = {
      {{{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT}, {1, BINDING_SAMPLER, 1, SHADER_VISIBILITY_FRAGMENT}}}};
  program_ = device_.CreateProgram(desc);
  if (!program_)
    return 1;
  const Vertex vertices[] = {{{0, .5f, 0}, {.5f, 0}}, {{-.5f, -.5f, 0}, {0, 1}}, {{.5f, -.5f, 0}, {1, 1}}};
  const uint32_t indices[] = {0, 1, 2};
  vertex_buffer_ = device_.CreateBuffer({sizeof(vertices)});
  if (!vertex_buffer_)
    return 1;
  index_buffer_ = device_.CreateBuffer({sizeof(indices)});
  if (!index_buffer_)
    return 1;
  grassland::rhix::Image texture = device_.CreateImage({256, 256, IMAGE_FORMAT_R8G8B8A8_UNORM});
  if (!texture)
    return 1;
  group_ = device_.CreateBindingGroup(desc.pipeline_layout.groups[0]);
  if (!group_)
    return 1;
  if (!group_.BindResource(0, texture))
    return 1;
  auto sampler = device_.CreateSampler();
  if (!sampler || !group_.BindResource(1, sampler))
    return 1;
  // Upload owns a snapshot and staging leases; later graphics work is ordered on the same native queue.
  {
    grassland::rhix::CommandContext upload = device_.CreateCommandContext();
    if (!upload)
      return 1;
    if (!upload.CmdUploadBuffer(vertex_buffer_, 0, vertices, sizeof(vertices)))
      return 1;
    if (!upload.CmdUploadBuffer(index_buffer_, 0, indices, sizeof(indices)))
      return 1;
    std::array<uint32_t, 256 * 256> pixels;
    for (uint32_t y = 0; y < 256; ++y)
      for (uint32_t x = 0; x < 256; ++x) {
        uint32_t v = x ^ y;
        pixels[y * 256 + x] = v | (v << 8) | (v << 16) | 0xff000000;
      }
    if (!upload.CmdUploadImage(texture, pixels.data(), 256 * 4))
      return 1;
    grassland::rhix::SubmissionToken uploaded = queue_.Submit(std::move(upload));
    if (!uploaded)
      return 1;
    if (headless_) {
      if (!uploaded.WaitHost())
        return 1;
    }
  }
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
  SetWarmupFrames(headless_ ? 60 : 0);

  initialized_ = true;
  return 0;
}

int TextureModule::OnUpdate() {
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

int TextureModule::OnRender() {
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
  grassland::rhix::Image image;
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
  grassland::rhix::CommandContext context = device_.CreateCommandContext();
  if (!context)
    return 1;
  if (!context.CmdBindProgram(program_))
    return 1;
  if (!context.CmdBindGroup(0, group_))
    return 1;
  if (!context.CmdBeginRendering(image, ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0.6f, 0.7f, 0.8f, 1}))
    return 1;
  if (!context.CmdBindVertexBuffer(0, vertex_buffer_))
    return 1;
  if (!context.CmdBindIndexBuffer(index_buffer_))
    return 1;
  if (!context.CmdSetViewport({0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1}))
    return 1;
  if (!context.CmdSetScissor({0, 0, extent.width, extent.height}))
    return 1;
  if (!context.CmdDrawIndexed(3))
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
  grassland::rhix::SubmissionToken token = queue_.Submit(std::move(context));

  if (!token)
    return 1;
  if (headless_)
    pending_[slot] = std::move(token);
  CompleteFrame();
  fps_.TickFrame();
  auto now = std::chrono::steady_clock::now();
  if (!headless_ && now - title_time_ >= std::chrono::milliseconds(250)) {
    if (!window_.SetTitle(fmt::format("RHIX texture - FPS: {:.1f} - P: toggle VSync", fps_.GetFPS())))
      return 1;
    title_time_ = now;
  }

  return 0;
}

int TextureModule::OnTerminate() {
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
            "texture: headless 1280x720, {} frames, {:.6f} s, {:.2f} FPS (3 in flight, 60 warmup)\n", GetFrameIndex(),
            seconds, GetFrameIndex() / seconds);
      }
      if (!capture_directory_.empty()) {
        if (!last_frame_ || !last_frame_.WaitHost())
          return 1;
        std::vector<uint8_t> pixels(last_frame_.GetSize());
        if (!last_frame_.CopyTo(pixels.data(), pixels.size()) ||
            !SaveImage(capture_directory_, "texture", pixels, last_frame_desc_, last_frame_space_,
                       last_frame_white_scale_))
          return 1;
      }
      return 0;
    }();
  }
  benchmark_start_ = {};
  title_time_ = {};
  fps_ = {};
  group_ = {};
  index_buffer_ = {};
  vertex_buffer_ = {};
  program_ = {};
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
