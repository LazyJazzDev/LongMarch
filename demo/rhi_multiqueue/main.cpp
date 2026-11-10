#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

#include "grassland/rhi/rhi.h"

namespace {
using namespace grassland::rhi;
constexpr uint32_t count = 65536;
constexpr uint32_t iterations = 32;
constexpr uint64_t bytes = uint64_t(count) * sizeof(uint32_t);

bool Run(IDevice *device, bool synchronize) {
  ObjPtr<IQueue> main = device->GetQueue(QUEUE_TYPE_MAIN);
  ObjPtr<IQueue> assist = device->GetQueue(QUEUE_TYPE_ASSIST);
  if (!main || !assist) {
    std::cerr << "Main and Assist queues are required.\n";
    return false;
  }
  ObjPtr<IBuffer> input = device->CreateBuffer({bytes});
  ObjPtr<IBuffer> output = device->CreateBuffer({bytes});
  // Extra independent copies make the missing dependency easier to observe.
  // They do not enforce an ordering between the queues.
  std::vector<uint32_t> padding(4 * 1024 * 1024, 0);
  ObjPtr<IBuffer> scratch = device->CreateBuffer({padding.size() * sizeof(uint32_t)});
  ObjPtr<IBuffer> scratch_copy = device->CreateBuffer({padding.size() * sizeof(uint32_t)});
  ObjPtr<IShader> shader = device->CreateShaderFromSource(R"(
    RHI_STRUCTURED_BUFFER(0, 0, uint, input);
    RHI_RW_STRUCTURED_BUFFER(0, 1, uint, output);
    [numthreads(64, 1, 1)]
    void CSMain(uint3 id : SV_DispatchThreadID) {
      output[id.x] = input[id.x] * 3u + 7u;
    }
  )",
                                                          "CSMain", SHADER_STAGE_COMPUTE);
  BindingLayoutDesc layout{{{0, BINDING_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_COMPUTE, 4},
                            {1, BINDING_RW_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_COMPUTE, 4}}};
  if (!input || !output || !scratch || !scratch_copy || !shader)
    return false;
  ComputeProgramDesc desc;
  desc.compute_shader = shader.Get();
  desc.pipeline_layout.groups = {layout};
  ObjPtr<IComputeProgram> program = device->CreateComputeProgram(desc);
  ObjPtr<IBindingGroup> group = device->CreateBindingGroup(layout);
  if (!program || !group || !group->BindResource(0, input.Get()) || !group->BindResource(1, output.Get()))
    return false;

  std::vector<uint32_t> data(count, 0), actual(count);
  ObjPtr<ICommandContext> initialize = assist->CreateCommandContext();
  if (!initialize || !initialize->CmdUploadBuffer(input.Get(), 0, data.data(), bytes))
    return false;
  ObjPtr<ISubmissionToken> initialized = initialize->Submit();
  if (!initialized || !initialized->WaitHost())
    return false;

  uint32_t incorrect_frames = 0;
  uint64_t incorrect_values = 0;
  for (uint32_t frame = 1; frame <= iterations; ++frame) {
    std::fill(data.begin(), data.end(), frame);
    ObjPtr<ICommandContext> update = assist->CreateCommandContext();
    ObjPtr<ICommandContext> compute = main->CreateCommandContext();
    if (!update || !compute)
      return false;
    if (!update->CmdUploadBuffer(scratch.Get(), 0, padding.data(), padding.size() * sizeof(uint32_t)))
      return false;
    for (int repeat = 0; repeat < 128; ++repeat)
      if (!update->CmdCopyBuffer(scratch.Get(), 0, scratch_copy.Get(), 0, padding.size() * sizeof(uint32_t)))
        return false;
    if (!update->CmdUploadBuffer(input.Get(), 0, data.data(), bytes) ||
        !compute->CmdBindComputeProgram(program.Get()) || !compute->CmdBindGroup(0, group.Get()) ||
        !compute->CmdDispatch(count / 64))
      return false;
    ObjPtr<IDownloadToken> download = compute->CmdDownloadBuffer(output.Get(), 0, bytes);
    if (!download)
      return false;
    ObjPtr<ISubmissionToken> uploaded = update->Submit();
    if (!uploaded)
      return false;
    SubmitInfo submit;
    if (synchronize)
      submit.waits.push_back(uploaded->GetEvent());
    ObjPtr<ISubmissionToken> computed = compute->Submit(submit);
    if (!computed || !computed->WaitHost() || !uploaded->WaitHost() || !download->CopyTo(actual.data(), bytes))
      return false;
    // Both queues finish before buffers are reused for the next iteration.
    const auto wrong =
        std::count_if(actual.begin(), actual.end(), [frame](uint32_t value) { return value != frame * 3u + 7u; });
    incorrect_frames += wrong != 0;
    incorrect_values += wrong;
  }
  std::cout << (synchronize ? "event" : "no-event") << ": incorrect frames " << incorrect_frames << "/" << iterations
            << ", incorrect values " << incorrect_values << "/" << uint64_t(count) * iterations << '\n';
  return !synchronize || incorrect_frames == 0;
}
}  // namespace

int main(int argc, char **argv) {
  using namespace grassland::rhi;
  if (argc != 3) {
    std::cerr << "Usage: demo_rhi_multiqueue <vulkan|d3d12|metal> <event|no-event>\n";
    return 2;
  }
  const std::string_view backend = argv[1], mode = argv[2];
  BackendAPI api;
  if (backend == "vulkan")
    api = BACKEND_API_VULKAN;
  else if (backend == "d3d12")
    api = BACKEND_API_D3D12;
  else if (backend == "metal")
    api = BACKEND_API_METAL;
  else
    return 2;
  if (mode != "event" && mode != "no-event")
    return 2;
  if (mode == "no-event")
    std::cout
        << "Intentionally missing synchronization: results are undefined; zero mismatches would not prove safety.\n";
  ObjPtr<ICore> core = CreateCore(api);
  if (!core)
    return 1;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  if (!device)
    return 1;
  std::cout << device->Name() << '\n';
  return Run(device.Get(), mode == "event") ? 0 : 1;
}
