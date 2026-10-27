#include "module.h"

#include <fmt/format.h>

#include <array>
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <map>
#include <string>
#include <thread>

#include "../../save_image.h"
#include "grassland/rhi/rhi.h"
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

struct TextureModule::State {
  bool initialized{};
  bool headless{};
  std::string capture_directory{};
  grassland::rhi::IDevice *device{};
  ObjPtr<IDownloadToken> last_frame{};
  ImageDesc last_frame_desc{};
  ColorSpace last_frame_space{};
  float last_frame_white_scale{};
  PresentMode requested{};
  bool present_mode_changed{};
  ObjPtr<IWindow> window{};
  ObjPtr<IQueue> queue{};
  std::array<ObjPtr<IImage>, 3> targets{};
  std::array<ObjPtr<ISubmissionToken>, 3> pending{};
  ObjPtr<IProgram> program{};
  ObjPtr<IBuffer> vertex_buffer{};
  ObjPtr<IBuffer> index_buffer{};
  ObjPtr<IBindingGroup> group{};
  grassland::FPSCounter fps{};
  std::chrono::steady_clock::time_point title_time{};
  std::chrono::steady_clock::time_point benchmark_start{};
};

TextureModule::TextureModule(grassland::rhi::ObjPtr<grassland::rhi::ICore> core,
                             grassland::rhi::ObjPtr<grassland::rhi::IDevice> device)
    : ModuleBase(std::move(core), std::move(device)) {
}

TextureModule::~TextureModule() = default;

int TextureModule::OnInit(const ModuleOptions &options) {
  state_ = std::make_unique<State>();

  state_->headless = options.headless;
  state_->capture_directory = options.capture_directory;
  using namespace grassland::rhi;
  state_->device = GetDevice();

  state_->last_frame_desc = ImageDesc{};
  state_->last_frame_space = COLOR_SPACE_SRGB_NONLINEAR;
  state_->last_frame_white_scale = 1.0f;
  state_->requested = PRESENT_MODE_VSYNC;
  state_->present_mode_changed = false;

  if (!state_->headless) {
    state_->window = state_->device->CreateGLFWWindow({1280, 720, "RHI texture - P: toggle VSync"});
    if (!state_->window)
      return 1;
  }
  state_->queue = state_->device->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!state_->queue)
    return 1;

  if (state_->headless) {
    for (auto &target : state_->targets) {
      target = state_->device->CreateImage({1280, 720, IMAGE_FORMAT_R8G8B8A8_UNORM});
      if (!target)
        return 1;
    }
  }
  const auto &embedded = shader_list.at("modules/texture/shaders/texture.slang");
  std::string source(embedded.first, embedded.second);
  ObjPtr<IShader> vertex_shader = state_->device->CreateShaderFromSource(source, "VSMain", SHADER_STAGE_VERTEX);
  if (!vertex_shader)
    return 1;
  ObjPtr<IShader> fragment_shader = state_->device->CreateShaderFromSource(source, "PSMain", SHADER_STAGE_FRAGMENT);
  if (!fragment_shader)
    return 1;
  ProgramDesc desc;
  desc.vertex_shader = vertex_shader.Get();
  desc.fragment_shader = fragment_shader.Get();
  desc.vertex_bindings = {{sizeof(Vertex), false}};
  desc.vertex_attributes = {{0, 0, INPUT_TYPE_FLOAT3, 0}, {1, 0, INPUT_TYPE_FLOAT2, 12}};
  {
    ObjPtr<IImage> image;
    if (state_->headless)
      image = state_->targets[0];
    else
      image = state_->window->GetNextImage();
    if (!image)
      return 1;
    desc.color_formats = {image->GetDesc().format};
  }
  desc.pipeline_layout.groups = {
      {{{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT}, {1, BINDING_SAMPLER, 1, SHADER_VISIBILITY_FRAGMENT}}}};
  state_->program = state_->device->CreateProgram(desc);
  if (!state_->program)
    return 1;
  const Vertex vertices[] = {{{0, .5f, 0}, {.5f, 0}}, {{-.5f, -.5f, 0}, {0, 1}}, {{.5f, -.5f, 0}, {1, 1}}};
  const uint32_t indices[] = {0, 1, 2};
  state_->vertex_buffer = state_->device->CreateBuffer({sizeof(vertices)});
  if (!state_->vertex_buffer)
    return 1;
  state_->index_buffer = state_->device->CreateBuffer({sizeof(indices)});
  if (!state_->index_buffer)
    return 1;
  ObjPtr<IImage> texture = state_->device->CreateImage({256, 256, IMAGE_FORMAT_R8G8B8A8_UNORM});
  if (!texture)
    return 1;
  state_->group = state_->device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  if (!state_->group)
    return 1;
  if (!state_->group->BindResource(0, texture.Get()))
    return 1;
  ObjPtr<ISampler> sampler = state_->device->CreateSampler();
  if (!sampler || !state_->group->BindResource(1, sampler.Get()))
    return 1;
  // Upload owns a snapshot and staging leases; later graphics work is ordered on the same native queue.
  {
    ObjPtr<ICommandContext> upload = state_->device->CreateCommandContext();
    if (!upload)
      return 1;
    if (!upload->CmdUploadBuffer(state_->vertex_buffer.Get(), 0, vertices, sizeof(vertices)))
      return 1;
    if (!upload->CmdUploadBuffer(state_->index_buffer.Get(), 0, indices, sizeof(indices)))
      return 1;
    std::array<uint32_t, 256 * 256> pixels;
    for (uint32_t y = 0; y < 256; ++y)
      for (uint32_t x = 0; x < 256; ++x) {
        uint32_t v = x ^ y;
        pixels[y * 256 + x] = v | (v << 8) | (v << 16) | 0xff000000;
      }
    if (!upload->CmdUploadImage(texture.Get(), pixels.data(), 256 * 4))
      return 1;
    ObjPtr<ISubmissionToken> uploaded = state_->queue->Submit(upload.Get());
    if (!uploaded)
      return 1;
    if (state_->headless) {
      if (!uploaded->WaitHost())
        return 1;
    }
  }
  if (!state_->headless) {
    state_->window->KeyEvent().RegisterCallback([this](int key, int, int action, int) {
      if (action != GLFW_PRESS)
        return;
      if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(state_->window->GLFWWindow(), GLFW_TRUE);
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

int TextureModule::OnUpdate() {
  if (!state_->headless) {
    if (state_->window->ShouldClose()) {
      RequestClose();
      return 0;
    }
    if (state_->window->IsSwapChainInvalid()) {
      if (!state_->window->RebuildSwapChain())
        return 1;
    }
    if (state_->present_mode_changed) {
      state_->present_mode_changed = false;
      if (!state_->window->SetPresentMode(state_->requested))
        return 1;
    }
  }

  return 0;
}

int TextureModule::OnRender() {
  const size_t slot = (static_cast<int64_t>(GetFrameIndex()) + 60) % state_->targets.size();
  if (state_->headless) {
    if (GetFrameIndex() == 0) {
      for (auto &token : state_->pending) {
        if (token) {
          if (!token->WaitHost())
            return 1;
          token = {};
        }
      }
      state_->benchmark_start = std::chrono::steady_clock::now();
    }
    if (state_->pending[slot]) {
      if (!state_->pending[slot]->WaitHost())
        return 1;
      state_->pending[slot] = {};
    }
  }
  ObjPtr<IImage> image;
  if (state_->headless)
    image = state_->targets[slot];
  else
    image = state_->window->GetNextImage();
  if (!image) {
    if (state_->headless)
      return 1;
    state_->fps.Reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return 0;
  }
  auto extent = image->GetDesc();
  ObjPtr<ICommandContext> context = state_->device->CreateCommandContext();
  if (!context)
    return 1;
  if (!context->CmdBindProgram(state_->program.Get()))
    return 1;
  if (!context->CmdBindGroup(0, state_->group.Get()))
    return 1;
  if (!context->CmdBeginRendering(
          {{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0.6f, 0.7f, 0.8f, 1}}}}))
    return 1;
  if (!context->CmdBindVertexBuffer(0, state_->vertex_buffer.Get()))
    return 1;
  if (!context->CmdBindIndexBuffer(state_->index_buffer.Get()))
    return 1;
  if (!context->CmdSetViewport({0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1}))
    return 1;
  if (!context->CmdSetScissor({0, 0, extent.width, extent.height}))
    return 1;
  if (!context->CmdDrawIndexed(3))
    return 1;
  if (!context->CmdEndRendering())
    return 1;
  if (!state_->headless) {
    if (!context->CmdPresent(state_->window.Get()))
      return 1;
  }
  if (!state_->capture_directory.empty()) {
    state_->last_frame = context->CmdDownloadImage(image.Get());
    if (!state_->last_frame)
      return 1;
    state_->last_frame_desc = image->GetDesc();
    state_->last_frame_space = state_->window ? state_->window->GetColorSpace() : COLOR_SPACE_SRGB_NONLINEAR;
    state_->last_frame_white_scale = state_->window ? state_->window->GetHDRReferenceWhiteScale() : 1.0f;
  }
  ObjPtr<ISubmissionToken> token = state_->queue->Submit(context.Get());

  if (!token)
    return 1;
  if (state_->headless)
    state_->pending[slot] = std::move(token);
  CompleteFrame();
  state_->fps.TickFrame();
  auto now = std::chrono::steady_clock::now();
  if (!state_->headless && now - state_->title_time >= std::chrono::milliseconds(250)) {
    if (!state_->window->SetTitle(fmt::format("RHI texture - FPS: {:.1f} - P: toggle VSync", state_->fps.GetFPS())))
      return 1;
    state_->title_time = now;
  }

  return 0;
}

int TextureModule::OnTerminate() {
  int result = 0;
  if (state_ && state_->initialized) {
    result = [&]() -> int {
      if (state_->headless) {
        for (const auto &token : state_->pending) {
          if (token) {
            if (!token->WaitHost())
              return 1;
          }
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - state_->benchmark_start).count();
        std::cout << fmt::format(
            "texture: headless 1280x720, {} frames, {:.6f} s, {:.2f} FPS (3 in flight, 60 warmup)\n", GetFrameIndex(),
            seconds, GetFrameIndex() / seconds);
      }
      if (!state_->capture_directory.empty()) {
        if (!state_->last_frame || !state_->last_frame->WaitHost())
          return 1;
        std::vector<uint8_t> pixels(state_->last_frame->GetSize());
        if (!state_->last_frame->CopyTo(pixels.data(), pixels.size()) ||
            !SaveImage(state_->capture_directory, "texture", pixels, state_->last_frame_desc, state_->last_frame_space,
                       state_->last_frame_white_scale))
          return 1;
      }
      return 0;
    }();
  }
  state_.reset();
  return result;
}
