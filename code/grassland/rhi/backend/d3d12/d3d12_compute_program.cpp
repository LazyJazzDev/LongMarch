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

ID3D12RootSignature *D3D12ComputeProgram::NativeRootSignature() const {
  return root_signature_.Get();
}

uint32_t D3D12ComputeProgram::RootParameter(uint32_t group, bool sampler) const {
  return root_parameters_[group][sampler];
}

bool D3D12ComputeProgram::Finalize() {
  if (finalized_)
    return true;
  if (!ValidateLayout())
    return false;
  std::vector<uint8_t> code;
  std::array<uint32_t, 3> threads{};
  if (!CompileSlang(desc_.compute_shader->GetDesc(), ShaderMacroHeader(desc_.pipeline_layout, BACKEND_API_D3D12),
                    BACKEND_API_D3D12, code, &desc_.pipeline_layout, &threads))
    return false;
  if (!threads[0] || !threads[1] || !threads[2] || threads[0] > 1024 || threads[1] > 1024 || threads[2] > 64 ||
      uint64_t(threads[0]) * threads[1] * threads[2] > 1024)
    return false;
  auto device = static_cast<D3D12Device *>(device_)->NativeDevice();
  ComPtr<ID3D12RootSignature> signature;
  if (!CreateRootSignature(device, desc_.pipeline_layout, false, signature, root_parameters_))
    return false;
  D3D12_COMPUTE_PIPELINE_STATE_DESC state{};
  state.pRootSignature = signature.Get();
  state.CS = {code.data(), code.size()};
  ComPtr<ID3D12PipelineState> pipeline;
  if (!Check(device->CreateComputePipelineState(&state, IID_PPV_ARGS(&pipeline)), "Create compute pipeline"))
    return false;
  root_signature_ = std::move(signature);
  pipeline_ = std::move(pipeline);
  finalized_ = true;
  return true;
}
}  // namespace grassland::rhi::backend::d3d12
