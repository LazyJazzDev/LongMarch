#include "../../module.h"

#include <fmt/format.h>

#include <chrono>
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

int HelloTriangle(grassland::rhi::BackendAPI api, int frames) {
  using namespace grassland::rhi;
  if (api == BACKEND_API_METAL) {
    std::cerr << "hello_triangle currently supports D3D12 and Vulkan.\n";
    return 1;
  }
  ObjPtr<ICore> core = CreateCore(api);
  if (!core)
    return 1;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  if (!device)
    return 1;
  ObjPtr<IWindow> window = device->CreateGLFWWindow({1280, 720, "RHI hello_triangle - P: toggle VSync"});
  if (!window)
    return 1;
  ObjPtr<IQueue> queue = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  if (!queue)
    return 1;
  const auto &embedded = shader_list.at("modules/hello_triangle/shaders/triangle.slang");
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
    ObjPtr<IImage> image = window->GetNextImage();
    if (!image)
      return 1;
    desc.color_format = image->GetDesc().format;
  }
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  if (!program)
    return 1;
  if (!program->Finalize())
    return 1;
  const Vertex vertices[] = {{{0, 0.6f, 0}, {1, 0, 0}}, {{-0.6f, -0.6f, 0}, {0, 0, 1}}, {{0.6f, -0.6f, 0}, {0, 1, 0}}};
  const uint32_t indices[] = {0, 1, 2};
  ObjPtr<IBuffer> vertex_buffer = device->CreateBuffer({sizeof(vertices)});
  if (!vertex_buffer)
    return 1;
  ObjPtr<IBuffer> index_buffer = device->CreateBuffer({sizeof(indices)});
  if (!index_buffer)
    return 1;
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
  }
  bool previous_p = false;
  PresentMode requested = PRESENT_MODE_VSYNC;
  grassland::FPSCounter fps;
  auto title_time = std::chrono::steady_clock::now();
  int rendered = 0;
  while (!frames || rendered < frames) {
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
    if (!context->CmdBindProgram(program.Get()))
      return 1;
    if (!context->CmdBeginRendering(
            {image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0.6f, 0.7f, 0.8f, 1}}))
      return 1;
    if (!context->CmdBindVertexBuffer(0, vertex_buffer.Get()))
      return 1;
    if (!context->CmdBindIndexBuffer(index_buffer.Get()))
      return 1;
    if (!context->CmdSetViewport({0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1}))
      return 1;
    if (!context->CmdSetScissor({0, 0, extent.width, extent.height}))
      return 1;
    if (!context->CmdDrawIndexed(3))
      return 1;
    if (!context->CmdEndRendering())
      return 1;
    if (!context->CmdPresent(window.Get()))
      return 1;
    ObjPtr<ISubmissionToken> token = queue->Submit(context.Get());
    if (!token)
      return 1;
    ++rendered;
    fps.TickFrame();
    auto now = std::chrono::steady_clock::now();
    if (now - title_time >= std::chrono::milliseconds(250)) {
      if (!window->SetTitle(fmt::format("RHI hello_triangle - FPS: {:.1f} - P: toggle VSync", fps.GetFPS())))
        return 1;
      title_time = now;
    }
  }
  return 0;
}
