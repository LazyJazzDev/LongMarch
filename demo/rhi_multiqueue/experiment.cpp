#include "experiment.h"

#include <array>

namespace multiqueue_demo {
using namespace grassland::rhi;

Result Run(BackendAPI api, bool synchronize, bool validation, uint32_t frames, uint32_t copies) {
  Result result;
  ObjPtr<ICore> core = CreateCore(api, {validation});
  if (!core)
    return result;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  if (!device)
    return result;
  result.device = device->Name();
  ObjPtr<IQueue> graphics = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  ObjPtr<IQueue> transfer = device->GetQueue(QUEUE_TYPE_TRANSFER);
  if (!graphics || !transfer)
    return result;
  result.distinct_queues = !graphics->SharesPhysicalQueue(transfer.Get());
  if (!result.distinct_queues)
    return result;
  const std::string shader = R"(
    struct Vertex { RHI_INPUT(0, float2, position); RHI_INPUT(1, float4, color); };
    struct Varying { float4 position : SV_Position; float4 color : COLOR0; };
    Varying VSMain(Vertex input) {
      Varying result; result.position = float4(input.position, 0, 1); result.color = input.color; return result;
    }
    float4 PSMain(Varying input) : SV_Target0 { return input.color; }
  )";
  ObjPtr<IShader> vs = device->CreateShaderFromSource(shader, "VSMain", SHADER_STAGE_VERTEX);
  ObjPtr<IShader> ps = device->CreateShaderFromSource(shader, "PSMain", SHADER_STAGE_FRAGMENT);
  if (!vs || !ps)
    return result;
  ProgramDesc desc;
  desc.vertex_shader = vs.Get();
  desc.fragment_shader = ps.Get();
  desc.vertex_bindings = {{24, false}};
  desc.vertex_attributes = {{0, 0, INPUT_TYPE_FLOAT2, 0}, {1, 0, INPUT_TYPE_FLOAT4, 8}};
  desc.rasterizer.cull_mode = CULL_MODE_NONE;
  ObjPtr<IProgram> program = device->CreateProgram(desc);
  ObjPtr<IImage> image = device->CreateImage({64, 64, IMAGE_FORMAT_R8G8B8A8_UNORM});
  ObjPtr<IBuffer> vertices = device->CreateBuffer({72});
  constexpr uint64_t backlog_size = 32 * 1024 * 1024;
  ObjPtr<IBuffer> scratch_a = device->CreateBuffer({backlog_size});
  ObjPtr<IBuffer> scratch_b = device->CreateBuffer({backlog_size});
  if (!program || !image || !vertices || !scratch_a || !scratch_b)
    return result;
  std::array<std::array<float, 6>, 3> data{{{{-1, -1, 0, 0, 0, 1}}, {{3, -1, 0, 0, 0, 1}}, {{-1, 3, 0, 0, 0, 1}}}};
  {
    std::vector<uint8_t> zero(backlog_size);
    ObjPtr<ICommandContext> init = graphics->CreateCommandContext();
    if (!init || !init->CmdUploadBuffer(vertices.Get(), 0, data.data(), sizeof(data)) ||
        !init->CmdUploadBuffer(scratch_a.Get(), 0, zero.data(), zero.size()))
      return result;
    ObjPtr<ISubmissionToken> done = init->Submit();
    if (!done || !done->WaitHost())
      return result;
  }
  for (uint32_t frame = 0; frame < frames; ++frame) {
    const auto channel = frame % 3;
    for (auto &vertex : data)
      for (uint32_t c = 0; c < 3; ++c)
        vertex[2 + c] = c == channel ? 1.0f : 0.0f;
    ObjPtr<ICommandContext> update = transfer->CreateCommandContext();
    if (!update)
      return result;
    // Delay the small update behind real GPU copies, without CPU sleeps or a host wait.
    for (uint32_t i = 0; i < copies; ++i) {
      if (!update->CmdCopyBuffer(scratch_a.Get(), 0, scratch_b.Get(), 0, backlog_size) ||
          !update->CmdCopyBuffer(scratch_b.Get(), 0, scratch_a.Get(), 0, backlog_size))
        return result;
    }
    if (!update->CmdUploadBuffer(vertices.Get(), 0, data.data(), sizeof(data)))
      return result;
    ObjPtr<ISubmissionToken> updated = update->Submit();
    if (!updated)
      return result;
    update.Release();
    ObjPtr<IEvent> ready = updated->GetEvent();
    ObjPtr<ICommandContext> render = graphics->CreateCommandContext();
    if (!render || !render->CmdBindProgram(program.Get()) || !render->CmdBindVertexBuffer(0, vertices.Get()) ||
        !render->CmdSetViewport({0, 0, 64, 64, 0, 1}) || !render->CmdSetScissor({0, 0, 64, 64}) ||
        !render->CmdBeginRendering(
            {{{image.Get(), ATTACHMENT_LOAD_OP_CLEAR, ATTACHMENT_STORE_OP_STORE, {0, 0, 0, 1}}}}) ||
        !render->CmdDraw(3) || !render->CmdEndRendering()) {
      updated->WaitHost();
      return result;
    }
    ObjPtr<IDownloadToken> download = render->CmdDownloadImage(image.Get());
    if (!download) {
      updated->WaitHost();
      return result;
    }
    SubmitInfo submit;
    if (synchronize)
      submit.waits = {ready.Get()};
    ObjPtr<ISubmissionToken> rendered = render->Submit(submit);
    render.Release();
    const bool rendered_ok = rendered && rendered->WaitHost();
    // Always join both queues before reusing data or destroying this frame's resources.
    const bool update_ok = updated->WaitHost();
    if (!rendered_ok || !update_ok)
      return result;
    result.pixels.resize(download->GetSize());
    if (!download->CopyTo(result.pixels.data(), result.pixels.size()))
      return result;
    for (size_t i = 0; i < result.pixels.size(); i += 4) {
      bool match = result.pixels[i + 3] == 255;
      for (uint32_t c = 0; c < 3; ++c)
        match &= result.pixels[i + c] == (c == channel ? 255 : 0);
      if (!match)
        ++result.mismatched_pixels;
    }
    ++result.frames;
  }
  result.success = true;
  return result;
}
}  // namespace multiqueue_demo
