#include "../../module.h"

#include <fmt/format.h>

#include <array>
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <map>
#include <string>
#include <thread>

#include "grassland/rhi/rhi.h"
#include "grassland/util/fps_counter.h"
#include "grassland/util/virtual_file_system.h"

namespace {
#include "built_in_shaders.inl"

struct Vertex {
  float position[3];
  float color[3];
};
}  // namespace

int RunCube(grassland::rhi::BackendAPI api, int frames, bool headless) {
  using namespace grassland::rhi;

  ObjPtr<ICore> core = CreateCore(api);
  if (!core)
    return 1;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  if (!device)
    return 1;
  PresentMode requested = PRESENT_MODE_VSYNC;
  bool present_mode_changed = false;
  ObjPtr<IWindow> window;
  if (!headless) {
    window = device->CreateGLFWWindow({1280, 720, "RHI cube - P: toggle VSync"});
    if (!window)
      return 1;
  }
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue)
    return 1;
  std::array<ObjPtr<IImage>, 3> targets;
  std::array<ObjPtr<ISubmissionToken>, 3> pending;
  if (headless) {
    for (auto &target : targets) {
      target = device->CreateImage({1280, 720, IMAGE_FORMAT_RGBA8_UNORM});
      if (!target)
        return 1;
    }
  }
  const auto &embedded = shader_list.at("modules/cube/shaders/cube.slang");
  std::string source(embedded.first, embedded.second);
  ObjPtr<IShader> vertex_shader = device->CreateShader({source, "VSMain", SHADER_STAGE_VERTEX});
  if (!vertex_shader)
    return 1;
  ObjPtr<IShader> fragment_shader = device->CreateShader({source, "PSMain", SHADER_STAGE_FRAGMENT});
  if (!fragment_shader)
    return 1;
  ProgramDesc desc;
  desc.vertex_shader = vertex_shader.Get();
  desc.fragment_shader = fragment_shader.Get();
  desc.vertex_bindings = {{sizeof(Vertex), false}};
  desc.vertex_attributes = {{0, 0, VERTEX_FORMAT_FLOAT3, 0}, {1, 0, VERTEX_FORMAT_FLOAT3, 12}};
  {
    ObjPtr<IImage> image;
    if (headless)
      image = targets[0];
    else
      image = window->GetNextImage();
    if (!image)
      return 1;
    desc.color_formats = {image->GetDesc().format};
  }
  desc.pipeline_layout.groups = {{{{0, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_VERTEX}}}};
  desc.depth_stencil.format = IMAGE_FORMAT_D32_FLOAT;
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
    ObjPtr<ICommandContext> upload = device->CreateCommandContext();
    if (!upload)
      return 1;
    if (!upload->CmdUploadBuffer(vertex_buffer.Get(), 0, vertices, sizeof(vertices)))
      return 1;
    if (!upload->CmdUploadBuffer(index_buffer.Get(), 0, indices, sizeof(indices)))
      return 1;
    ObjPtr<ISubmissionToken> uploaded = queue->Submit(upload.Get());
    if (!uploaded)
      return 1;
    if (headless) {
      if (!uploaded->WaitHost())
        return 1;
    }
  }
  if (!headless) {
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
  }
  grassland::FPSCounter fps;
  auto title_time = std::chrono::steady_clock::now();
  // Warm up allocations and pipelines before measuring completed GPU work.
  auto benchmark_start = std::chrono::steady_clock::now();
  int rendered = headless ? -60 : 0;
  while (!frames || rendered < frames) {
    if (!headless) {
      if (!PollEvents())
        break;
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
    // GPU copies and draws are ordered on the same queue; the upload takes a CPU snapshot.
    const float angle =
        headless ? float(rendered + 60) * .01f
                 : float(std::chrono::duration<double>(std::chrono::steady_clock::now() - benchmark_start).count());

    struct Uniform {
      glm::mat4 model, view, projection;
    } ubo;

    ubo.model = glm::rotate(glm::mat4(1), angle, glm::vec3(0, 1, 0));
    ubo.view = glm::lookAt(glm::vec3(0, 0, 5), glm::vec3(0), glm::vec3(0, 1, 0));
    ubo.projection = glm::perspectiveZO(glm::radians(45.0f), float(extent.width) / extent.height, 3.5f, 6.5f);
    if (!context->CmdUploadBuffer(uniform.Get(), 0, &ubo, sizeof(ubo)))
      return 1;
    if (!depth || depth->GetDesc().width != extent.width || depth->GetDesc().height != extent.height) {
      depth = device->CreateImage({extent.width, extent.height, IMAGE_FORMAT_D32_FLOAT});
      if (!depth)
        return 1;
    }
    if (!context->CmdBindProgram(program.Get()))
      return 1;
    if (!context->CmdBindGroup(0, group.Get()))
      return 1;
    grassland::rhi::RenderingInfo render;
    render.color_attachments = {{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {.6f, .7f, .8f, 1}}};
    render.depth_stencil.image = depth.Get();
    render.depth_stencil.depth_load_op = ATTACHMENT_LOAD_OP_CLEAR;
    if (!context->CmdBeginRendering(render))
      return 1;
    if (!context->CmdBindVertexBuffer(0, vertex_buffer.Get()))
      return 1;
    if (!context->CmdBindIndexBuffer(index_buffer.Get()))
      return 1;
    if (!context->CmdSetViewport({0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1}))
      return 1;
    if (!context->CmdSetScissor({0, 0, extent.width, extent.height}))
      return 1;
    if (!context->CmdDrawIndexed(36))
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
      if (!window->SetTitle(fmt::format("RHI cube - FPS: {:.1f} - P: toggle VSync", fps.GetFPS())))
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
    std::cout << fmt::format("cube: headless 1280x720, {} frames, {:.6f} s, {:.2f} FPS (3 in flight, 60 warmup)\n",
                             rendered, seconds, rendered / seconds);
  }
  return 0;
}
