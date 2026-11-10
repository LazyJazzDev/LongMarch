#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "grassland/rhi/rhi.h"

using namespace grassland::rhi;

namespace {
using Clock = std::chrono::steady_clock;

ObjPtr<ICommandContext> Context(IDevice *device, IQueue *queue) {
#ifdef RHI_BENCH_LEGACY
  return device->CreateCommandContext();
#else
  return queue->CreateCommandContext();
#endif
}

ObjPtr<ISubmissionToken> Submit(IQueue *queue, ICommandContext *context, ISubmissionToken *dependency = nullptr) {
  SubmitInfo info;
  if (dependency) {
#ifdef RHI_BENCH_LEGACY
    info.waits.push_back(dependency);
#else
    info.waits.push_back(dependency->GetEvent());
#endif
  }
#ifdef RHI_BENCH_LEGACY
  return queue->Submit(context, info);
#else
  return context->Submit(info);
#endif
}

struct Slot {
  ObjPtr<IBuffer> input, output;
  ObjPtr<IBottomLevelAccelerationStructure> blas;
  ObjPtr<ITopLevelAccelerationStructure> tlas;
  ObjPtr<IBindingGroup> bindings;
  ObjPtr<ISubmissionToken> done;
  uint32_t parity{};
};

uint32_t Compute(uint32_t v, uint32_t loops) {
  for (uint32_t i = 0; i < loops; ++i)
    v = (v ^ (v >> 13)) * 1664525u + 1013904223u;
  return v;
}
}  // namespace

// Fixed work, no window, no synthetic GPU delays, no readback inside the timed loop.
int main(int argc, char **argv) {
  if (argc != 9) {
    std::cerr << "Usage: demo_rhi_queue_bench <vulkan|d3d12> <main|main-batched|assist> <upload|as-build|as-update> "
                 "<elements-or-grid-side> <compute-loops-or-rays> <inflight> <frames> <validation:0|1>\n";
    return 2;
  }
  const std::string api_name = argv[1], mode = argv[2], workload = argv[3];
  if ((api_name != "vulkan" && api_name != "d3d12") || (mode != "main" && mode != "main-batched" && mode != "assist") ||
      (workload != "upload" && workload != "as-build" && workload != "as-update"))
    return 2;
  const uint32_t size = std::stoul(argv[4]), loops = std::stoul(argv[5]), depth = std::stoul(argv[6]),
                 frames = std::stoul(argv[7]);
  const bool validate = std::string(argv[8]) == "1", ray = workload != "upload";
  if (!size || !loops || !depth || !frames || (!ray && size % 64))
    return 2;
  ObjPtr<ICore> core = CreateCore(api_name == "vulkan" ? BACKEND_API_VULKAN : BACKEND_API_D3D12, {validate});
  if (!core)
    return 1;
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  if (!device || (ray && !device->SupportRayQuery()))
    return 1;
#ifdef RHI_BENCH_LEGACY
  if (mode == "assist")
    return 2;
  ObjPtr<IQueue> main = device->GetQueue(QUEUE_TYPE_GRAPHICS);
  const char *revision = "baseline";
#else
  ObjPtr<IQueue> main = device->GetQueue(QUEUE_TYPE_MAIN);
  const char *revision = "current";
#endif
  ObjPtr<IQueue> assist;
#ifndef RHI_BENCH_LEGACY
  if (mode == "assist")
    assist = device->GetQueue(QUEUE_TYPE_ASSIST);
#endif
  IQueue *prep = mode == "assist" ? assist.Get() : main.Get();
  if (!main || !prep)
    return 1;
  const uint32_t elements = ray ? 512 * 512 : size;
  const uint64_t output_bytes = uint64_t(elements) * 4;
  std::array<std::vector<uint32_t>, 2> inputs;
  std::array<std::vector<float>, 2> vertices;
  uint64_t input_bytes;
  if (ray) {
    for (uint32_t parity = 0; parity < 2; ++parity) {
      auto &v = vertices[parity];
      v.reserve(uint64_t(size) * size * 18);
      for (uint32_t y = 0; y < size; ++y)
        for (uint32_t x = 0; x < size; ++x) {
          const float x0 = float(x) / size, x1 = float(x + 1) / size, y0 = float(y) / size, y1 = float(y + 1) / size;
          const float z = float(parity) * 0.125f;
          v.insert(v.end(), {x0, y0, z, x1, y0, z, x0, y1, z, x0, y1, z, x1, y0, z, x1, y1, z});
        }
    }
    input_bytes = vertices[0].size() * sizeof(float);
  } else {
    for (uint32_t parity = 0; parity < 2; ++parity) {
      inputs[parity].resize(size);
      for (uint32_t i = 0; i < size; ++i)
        inputs[parity][i] = i + parity * 100003u;
    }
    input_bytes = uint64_t(size) * 4;
  }
  std::string shader = "static const uint LOOPS = " + std::to_string(loops) + ";\n";
  shader += ray ? R"(
RHI_ACCELERATION_STRUCTURE(0, 0, scene);
RHI_RW_STRUCTURED_BUFFER(0, 1, uint, output);
[numthreads(64,1,1)] void CSMain(uint3 id : SV_DispatchThreadID) {
  uint sum = 0;
  for (uint i=0; i<LOOPS; ++i) {
    RayDesc r;
    r.Origin = float3((float(id.x % 512) + 0.25 + float(i % 4)*0.1)/512.0,
                     (float(id.x / 512) + 0.3)/512.0, -1.0);
    r.Direction = float3(0,0,1); r.TMin=0.0; r.TMax=2.0;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 255, r);
    while(q.Proceed()) {}
    sum += q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? uint(round(q.CommittedRayT()*1024.0)) : 0;
  }
  output[id.x] = sum;
}
)"
                : R"(
RHI_STRUCTURED_BUFFER(0, 0, uint, input);
RHI_RW_STRUCTURED_BUFFER(0, 1, uint, output);
[numthreads(64,1,1)] void CSMain(uint3 id : SV_DispatchThreadID) {
  uint v=input[id.x];
  for(uint i=0; i<LOOPS; ++i) v=(v^(v>>13))*1664525u+1013904223u;
  output[id.x]=v;
}
)";
  ObjPtr<IShader> cs = device->CreateShaderFromSource(shader, "CSMain", SHADER_STAGE_COMPUTE);
  if (!cs)
    return 1;
  BindingLayoutDesc layout{{{0, ray ? BINDING_ACCELERATION_STRUCTURE : BINDING_STRUCTURED_BUFFER, 1,
                             SHADER_VISIBILITY_COMPUTE, ray ? 0u : 4u},
                            {1, BINDING_RW_STRUCTURED_BUFFER, 1, SHADER_VISIBILITY_COMPUTE, 4}}};
  ComputeProgramDesc pd;
  pd.compute_shader = cs.Get();
  pd.pipeline_layout.groups = {layout};
  ObjPtr<IComputeProgram> program = device->CreateComputeProgram(pd);
  if (!program)
    return 1;
  std::vector<Slot> slots(depth);
  for (auto &s : slots) {
    s.input = device->CreateBuffer({input_bytes});
    s.output = device->CreateBuffer({output_bytes});
    s.bindings = device->CreateBindingGroup(layout);
    if (!s.input || !s.output || !s.bindings)
      return 1;
    if (ray) {
      BottomLevelAccelerationStructureDesc bd;
      bd.geometries = {{size * size * 6, 12, 0}};
      bd.allow_update = workload == "as-update";
      s.blas = device->CreateBottomLevelAccelerationStructure(bd);
      s.tlas = device->CreateTopLevelAccelerationStructure({1, workload == "as-update"});
      if (!s.blas || !s.tlas || !s.bindings->BindResource(0, s.tlas.Get()))
        return 1;
    } else if (!s.bindings->BindResource(0, s.input.Get()))
      return 1;
    if (!s.bindings->BindResource(1, s.output.Get()))
      return 1;
  }
  auto run = [&](uint32_t n, bool warmup) {
    for (uint32_t frame = 0; frame < n; ++frame) {
      auto &s = slots[frame % depth];
      if (s.done && !s.done->WaitHost())
        return false;
      s.parity = (frame / depth + frame) % 2;
      auto update = Context(device.Get(), prep);
      const void *data = ray ? static_cast<const void *>(vertices[s.parity].data()) : inputs[s.parity].data();
      if (!update || !update->CmdUploadBuffer(s.input.Get(), 0, data, input_bytes))
        return false;
      if (ray) {
        const auto build_mode = workload == "as-update" && (!warmup || frame >= depth) ? ACCELERATION_STRUCTURE_UPDATE
                                                                                       : ACCELERATION_STRUCTURE_BUILD;
        if (!update->CmdBuildBottomLevelAccelerationStructure(s.blas.Get(), {{{s.input.Get()}, {}}}, build_mode))
          return false;
        RayTracingInstance instance;
        instance.blas = s.blas.Get();
        if (!update->CmdBuildTopLevelAccelerationStructure(s.tlas.Get(), {instance}, build_mode))
          return false;
      }
      ObjPtr<ISubmissionToken> prepared;
      if (mode != "main-batched") {
        prepared = Submit(prep, update.Get());
        if (!prepared)
          return false;
      }
      auto compute =
          mode == "main-batched" ? ObjPtr<ICommandContext>::Retain(update.Get()) : Context(device.Get(), main.Get());
      if (!compute || !compute->CmdBindComputeProgram(program.Get()) || !compute->CmdBindGroup(0, s.bindings.Get()) ||
          !compute->CmdDispatch(elements / 64))
        return false;
      s.done = Submit(main.Get(), compute.Get(), mode == "assist" ? prepared.Get() : nullptr);
      if (!s.done)
        return false;
    }
    for (auto &s : slots)
      if (s.done && !s.done->WaitHost())
        return false;
    return true;
  };
  if (!run(std::max(32u, depth * 2), true))
    return 1;
  const auto begin = Clock::now();
  if (!run(frames, false))
    return 1;
  const double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
  uint64_t mismatches = 0;
  std::vector<uint32_t> actual(elements);
  for (auto &s : slots) {
    auto read = Context(device.Get(), main.Get());
    if (!read)
      return 1;
    ObjPtr<IDownloadToken> download = read->CmdDownloadBuffer(s.output.Get(), 0, output_bytes);
    auto done = Submit(main.Get(), read.Get());
    if (!download || !done || !done->WaitHost() || !download->CopyTo(actual.data(), output_bytes))
      return 1;
    for (uint32_t i = 0; i < elements; ++i) {
      const uint32_t expected = ray ? (1024 + s.parity * 128) * loops : Compute(inputs[s.parity][i], loops);
      mismatches += actual[i] != expected;
    }
  }
  std::cout << std::setprecision(9) << "RESULT," << revision << ',' << api_name << ',' << mode << ',' << workload << ','
            << size << ',' << loops << ',' << depth << ',' << frames << ',' << seconds << ',' << frames / seconds << ','
            << mismatches << ',' << device->Name() << '\n';
  return mismatches ? 1 : 0;
}
