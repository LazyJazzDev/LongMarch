#include "grassland/rhi/backend/d3d12/d3d12_compute_program.h"

#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
D3D12ComputeProgram::D3D12ComputeProgram(D3D12Device *device, const ComputeProgramDesc &desc)
    : ComputeProgram(device, desc) {
}

ID3D12PipelineState *D3D12ComputeProgram::NativePipeline() const {
  return pipeline_.Get();
}

D3D12ComputeProgram *D3D12ComputeProgram::Create(D3D12Device *owner_device, const ComputeProgramDesc &desc) {
  auto object = std::unique_ptr<D3D12ComputeProgram>(new D3D12ComputeProgram(owner_device, desc));
  if (!object->ValidateLayout())
    return nullptr;
  std::vector<uint8_t> code;
  std::array<uint32_t, 3> threads{};
  if (!CompileSlang(ShaderCompileDesc(object->desc_.compute_shader),
                    ShaderMacroHeader(object->desc_.pipeline_layout, BACKEND_API_D3D12), BACKEND_API_D3D12, code,
                    &object->desc_.pipeline_layout, &threads))
    return nullptr;
  if (!threads[0] || !threads[1] || !threads[2] || threads[0] > 1024 || threads[1] > 1024 || threads[2] > 64 ||
      uint64_t(threads[0]) * threads[1] * threads[2] > 1024)
    return nullptr;
  auto device = owner_device->NativeDevice();
  if (!object->CreateLayout(device, object->desc_.pipeline_layout, D3D12_ROOT_SIGNATURE_FLAG_NONE))
    return nullptr;
  D3D12_COMPUTE_PIPELINE_STATE_DESC state{};
  state.pRootSignature = object->NativeRootSignature();
  state.CS = {code.data(), code.size()};
  ComPtr<ID3D12PipelineState> pipeline;
  if (!Check(device->CreateComputePipelineState(&state, IID_PPV_ARGS(&pipeline)), "Create compute pipeline"))
    return nullptr;
  object->pipeline_ = std::move(pipeline);
  return object.release();
}
}  // namespace grassland::rhi::backend::d3d12
