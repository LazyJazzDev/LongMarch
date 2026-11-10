#include <array>
#include <chrono>
#include <string>
#include <thread>

#include "grassland/rhi/rhi.h"

// Evaluate once and stop recording immediately when a command fails.
#define RETURN_IF_FAIL(expression) \
  do {                             \
    if (!(expression))             \
      return 1;                    \
  } while (false)

namespace {
#include "shaders/texture.slang.h"

struct Vertex {
  float position[3];
  float uv[2];
};
}  // namespace

int RunTexture(grassland::rhi::IDevice *device) {
  using namespace grassland::rhi;

  ObjPtr<IWindow> window = device->CreateGLFWWindow({1280, 720, "RHI texture - Escape: close"});
  if (!window)
    return 1;

  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  if (!queue)
    return 1;

  std::string source(reinterpret_cast<const char *>(shaders_texture_slang), shaders_texture_slang_len);
  ObjPtr<IShader> vertex_shader = device->CreateShaderFromSource(source, "VSMain", SHADER_STAGE_VERTEX);
  if (!vertex_shader)
    return 1;
  ObjPtr<IShader> fragment_shader = device->CreateShaderFromSource(source, "PSMain", SHADER_STAGE_FRAGMENT);
  if (!fragment_shader)
    return 1;
  ProgramDesc desc;
  desc.vertex_shader = vertex_shader.Get();
  desc.fragment_shader = fragment_shader.Get();
  desc.vertex_bindings = {{sizeof(Vertex), false}};
  desc.vertex_attributes = {{0, 0, INPUT_TYPE_FLOAT3, 0}, {1, 0, INPUT_TYPE_FLOAT2, 12}};
  {
    ObjPtr<IImage> image = window->GetNextImage();
    if (!image)
      return 1;
    desc.color_formats = {image->GetDesc().format};
  }
  desc.pipeline_layout.groups = {
      {{{0, BINDING_TEXTURE2D, 1, SHADER_VISIBILITY_FRAGMENT}, {1, BINDING_SAMPLER, 1, SHADER_VISIBILITY_FRAGMENT}}}};
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  if (!program)
    return 1;
  const Vertex vertices[] = {{{0, .5f, 0}, {.5f, 0}}, {{-.5f, -.5f, 0}, {0, 1}}, {{.5f, -.5f, 0}, {1, 1}}};
  const uint32_t indices[] = {0, 1, 2};
  ObjPtr<IBuffer> vertex_buffer = device->CreateBuffer({sizeof(vertices)});
  if (!vertex_buffer)
    return 1;
  ObjPtr<IBuffer> index_buffer = device->CreateBuffer({sizeof(indices)});
  if (!index_buffer)
    return 1;
  ObjPtr<IImage> texture = device->CreateImage({256, 256, IMAGE_FORMAT_R8G8B8A8_UNORM});
  if (!texture)
    return 1;
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  if (!group)
    return 1;
  if (!group->BindResource(0, texture.Get()))
    return 1;
  ObjPtr<ISampler> sampler = device->CreateSampler();
  if (!sampler || !group->BindResource(1, sampler.Get()))
    return 1;
  // Upload owns a snapshot and staging leases; later graphics work is ordered on the same native queue.
  {
    ObjPtr<ICommandContext> upload = device->CreateCommandContext();
    if (!upload)
      return 1;
    RETURN_IF_FAIL(upload->CmdUploadBuffer(vertex_buffer.Get(), 0, vertices, sizeof(vertices)));
    RETURN_IF_FAIL(upload->CmdUploadBuffer(index_buffer.Get(), 0, indices, sizeof(indices)));
    std::array<uint32_t, 256 * 256> pixels;
    for (uint32_t y = 0; y < 256; ++y)
      for (uint32_t x = 0; x < 256; ++x) {
        uint32_t v = x ^ y;
        pixels[y * 256 + x] = v | (v << 8) | (v << 16) | 0xff000000;
      }
    RETURN_IF_FAIL(upload->CmdUploadImage(texture.Get(), pixels.data(), 256 * 4));
    ObjPtr<ISubmissionToken> uploaded = queue->Submit(upload.Get());
    if (!uploaded)
      return 1;
  }

  window->KeyEvent().RegisterCallback([&](int key, int, int action, int) {
    if (action != GLFW_PRESS)
      return;
    if (key == GLFW_KEY_ESCAPE)
      glfwSetWindowShouldClose(window->GLFWWindow(), GLFW_TRUE);
  });

  while (true) {
    if (!PollEvents())
      return 1;
    if (window->ShouldClose())
      break;
    if (window->IsSwapChainInvalid()) {
      if (!window->RebuildSwapChain())
        return 1;
    }

    ObjPtr<IImage> image = window->GetNextImage();
    if (!image) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    auto extent = image->GetDesc();
    ObjPtr<ICommandContext> context = device->CreateCommandContext();
    if (!context)
      return 1;
    RETURN_IF_FAIL(context->CmdBindProgram(program.Get()));
    RETURN_IF_FAIL(context->CmdBindGroup(0, group.Get()));
    RETURN_IF_FAIL(context->CmdBeginRendering(
        {{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0.6f, 0.7f, 0.8f, 1}}}}));
    RETURN_IF_FAIL(context->CmdBindVertexBuffer(0, vertex_buffer.Get()));
    RETURN_IF_FAIL(context->CmdBindIndexBuffer(index_buffer.Get()));
    RETURN_IF_FAIL(
        context->CmdSetViewport({0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1}));
    RETURN_IF_FAIL(context->CmdSetScissor({0, 0, extent.width, extent.height}));
    RETURN_IF_FAIL(context->CmdDrawIndexed(3));
    RETURN_IF_FAIL(context->CmdEndRendering());

    RETURN_IF_FAIL(context->CmdPresent(window.Get()));

    ObjPtr<ISubmissionToken> token = queue->Submit(context.Get());

    if (!token)
      return 1;
  }

  return 0;
}

#undef RETURN_IF_FAIL
