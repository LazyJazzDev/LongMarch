#include <chrono>
#include <glm/gtc/matrix_transform.hpp>
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
#include "shaders/cube.slang.h"

struct Vertex {
  float position[3];
  float color[3];
};
}  // namespace

int RunCube(grassland::rhi::IDevice *device) {
  using namespace grassland::rhi;

  ObjPtr<IWindow> window = device->CreateGLFWWindow({1280, 720, "RHI cube - Escape: close"});
  if (!window)
    return 1;

  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_MAIN);
  if (!queue)
    return 1;

  std::string source(reinterpret_cast<const char *>(shaders_cube_slang), shaders_cube_slang_len);
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
  desc.vertex_attributes = {{0, 0, INPUT_TYPE_FLOAT3, 0}, {1, 0, INPUT_TYPE_FLOAT3, 12}};
  {
    ObjPtr<IImage> image = window->GetNextImage();
    if (!image)
      return 1;
    desc.color_formats = {image->GetDesc().format};
  }
  desc.pipeline_layout.groups = {{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_VERTEX}}}};
  desc.depth_stencil.format = IMAGE_FORMAT_D32_SFLOAT;
  desc.depth_stencil.depth_test = true;
  desc.depth_stencil.depth_write = true;
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  if (!program)
    return 1;
  const Vertex vertices[] = {{{-1, -1, 1}, {0, 0, 0}}, {{1, -1, 1}, {1, 0, 0}},   {{-1, 1, 1}, {0, 1, 0}},
                             {{1, 1, 1}, {1, 1, 0}},   {{-1, -1, -1}, {0, 0, 1}}, {{1, -1, -1}, {1, 0, 1}},
                             {{-1, 1, -1}, {0, 1, 1}}, {{1, 1, -1}, {1, 1, 1}}};
  const uint32_t indices[] = {0, 1, 2, 2, 1, 3, 2, 3, 6, 6, 3, 7, 6, 7, 4, 4, 7, 5,
                              4, 5, 0, 0, 5, 1, 1, 5, 3, 3, 5, 7, 0, 2, 4, 4, 2, 6};
  ObjPtr<IBuffer> vertex_buffer = device->CreateBuffer({sizeof(vertices)});
  if (!vertex_buffer)
    return 1;
  ObjPtr<IBuffer> index_buffer = device->CreateBuffer({sizeof(indices)});
  if (!index_buffer)
    return 1;
  ObjPtr<IBuffer> uniform = device->CreateBuffer({256});
  if (!uniform)
    return 1;
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(desc.pipeline_layout.groups[0]);
  if (!group)
    return 1;
  if (!group->BindResource(0, uniform.Get()))
    return 1;
  ObjPtr<IImage> depth;
  // Upload owns a snapshot and staging leases; later graphics work is ordered on the same native queue.
  {
    ObjPtr<ICommandContext> upload = queue->CreateCommandContext();
    if (!upload)
      return 1;
    RETURN_IF_FAIL(upload->CmdUploadBuffer(vertex_buffer.Get(), 0, vertices, sizeof(vertices)));
    RETURN_IF_FAIL(upload->CmdUploadBuffer(index_buffer.Get(), 0, indices, sizeof(indices)));
    ObjPtr<ISubmissionToken> uploaded = upload->Submit();
    if (!uploaded)
      return 1;
  }

  window->KeyEvent().RegisterCallback([&](int key, int, int action, int) {
    if (action != GLFW_PRESS)
      return;
    if (key == GLFW_KEY_ESCAPE)
      glfwSetWindowShouldClose(window->GLFWWindow(), GLFW_TRUE);
  });

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

    ObjPtr<IImage> image = window->GetNextImage();
    if (!image) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    auto extent = image->GetDesc();
    ObjPtr<ICommandContext> context = queue->CreateCommandContext();
    if (!context)
      return 1;

    // GPU copies and draws are ordered on the same queue; the upload takes a CPU snapshot.
    const float angle = std::chrono::duration<float>(std::chrono::steady_clock::now() - animation_start).count();

    struct Uniform {
      glm::mat4 model, view, projection;
    } ubo;

    ubo.model = glm::rotate(glm::mat4(1), angle, glm::vec3(0, 1, 0));
    ubo.view = glm::lookAt(glm::vec3(0, 0, 5), glm::vec3(0), glm::vec3(0, 1, 0));
    ubo.projection = glm::perspectiveZO(glm::radians(45.0f), float(extent.width) / extent.height, 3.5f, 6.5f);
    RETURN_IF_FAIL(context->CmdUploadBuffer(uniform.Get(), 0, &ubo, sizeof(ubo)));
    if (!depth || depth->GetDesc().width != extent.width || depth->GetDesc().height != extent.height) {
      depth = device->CreateImage({extent.width, extent.height, IMAGE_FORMAT_D32_SFLOAT});
      if (!depth)
        return 1;
    }
    RETURN_IF_FAIL(context->CmdBindProgram(program.Get()));
    RETURN_IF_FAIL(context->CmdBindGroup(0, group.Get()));
    grassland::rhi::RenderingInfo render;
    render.color_attachments = {{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {.6f, .7f, .8f, 1}}};
    render.depth_stencil.image = depth.Get();
    render.depth_stencil.depth_load_op = ATTACHMENT_LOAD_OP_CLEAR;
    RETURN_IF_FAIL(context->CmdBeginRendering(render));
    RETURN_IF_FAIL(context->CmdBindVertexBuffer(0, vertex_buffer.Get()));
    RETURN_IF_FAIL(context->CmdBindIndexBuffer(index_buffer.Get()));
    RETURN_IF_FAIL(
        context->CmdSetViewport({0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1}));
    RETURN_IF_FAIL(context->CmdSetScissor({0, 0, extent.width, extent.height}));
    RETURN_IF_FAIL(context->CmdDrawIndexed(36));
    RETURN_IF_FAIL(context->CmdEndRendering());

    RETURN_IF_FAIL(context->CmdPresent(window.Get()));

    ObjPtr<ISubmissionToken> token = context->Submit();

    if (!token)
      return 1;
  }

  return 0;
}

#undef RETURN_IF_FAIL
