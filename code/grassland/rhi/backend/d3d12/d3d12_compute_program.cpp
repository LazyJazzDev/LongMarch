#include "grassland/rhi/backend/d3d12/d3d12_compute_program.h"

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/d3d12/d3d12_compute_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
D3D12ComputeProgramImpl::D3D12ComputeProgramImpl(ID3D12Device *device, const ComputeProgramDesc &desc) {
  InitializeComputeProgram(device, desc);
}

ID3D12PipelineState *D3D12ComputeProgramImpl::NativePipeline() const {
  return pipeline_.Get();
}

rhi::IComputeProgram *D3D12ComputeProgramImpl::Create(ID3D12Device *owner_device, const ComputeProgramDesc &desc) {
  auto object = std::unique_ptr<D3D12ComputeProgramImpl>(new D3D12ComputeProgramImpl(owner_device, desc));
  if (!object->ValidateLayout())
    return nullptr;
  std::vector<uint8_t> code;
  std::array<uint32_t, 3> threads{};
  if (!CompileSlang(object->desc_.compute_shader, ShaderMacroHeader(object->desc_.pipeline_layout, BACKEND_API_D3D12),
                    BACKEND_API_D3D12, code, &object->desc_.pipeline_layout, &threads))
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

rhi::IComputeProgram *ID3D12ComputeProgram::Create(ID3D12Device *device, const ComputeProgramDesc &desc) {
  return D3D12ComputeProgramImpl::Create(device, desc);
}

void D3D12ComputeProgramImpl::InitializeComputeProgram(ID3D12Device *device, const ComputeProgramDesc &desc) {
  device_ = device;
  desc_ = desc;
  shader_ = ObjectPtr<IShader>::Retain(desc.compute_shader);
}

const ComputeProgramDesc &D3D12ComputeProgramImpl::GetDesc() const {
  return desc_;
}

ID3D12Device *D3D12ComputeProgramImpl::Owner() const {
  return device_;
}

bool D3D12ComputeProgramImpl::ValidateLayout() const {
  if (!desc_.compute_shader || desc_.compute_shader->GetStage() != SHADER_STAGE_COMPUTE ||
      desc_.pipeline_layout.groups.size() > 8)
    return false;
  for (const auto &group : desc_.pipeline_layout.groups)
    if (!ValidateBindingLayout(group))
      return false;
  return true;
}

ID3D12RootSignature *D3D12ComputeProgramImpl::NativeRootSignature() const {
  return root_signature_.Get();
}

uint32_t D3D12ComputeProgramImpl::RootParameter(uint32_t group, bool sampler) const {
  return root_parameters_[group][sampler];
}

bool D3D12ComputeProgramImpl::CreateLayout(::ID3D12Device *device,
                                           const PipelineLayoutDesc &layout,
                                           D3D12_ROOT_SIGNATURE_FLAGS flags) {
  D3D12_ROOT_SIGNATURE_DESC root{};
  root.Flags = flags;
  const auto mappings = MapBindings(layout, BACKEND_API_D3D12);
  std::vector<std::vector<D3D12_DESCRIPTOR_RANGE>> ranges(layout.groups.size() * 2);
  for (auto m : mappings) {
    D3D12_DESCRIPTOR_RANGE range{};
    range.RangeType = m.desc.type == BINDING_SAMPLER          ? D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER
                      : m.desc.type == BINDING_UNIFORM_BUFFER ? D3D12_DESCRIPTOR_RANGE_TYPE_CBV
                      : (m.desc.type == BINDING_RW_STRUCTURED_BUFFER || m.desc.type == BINDING_RW_BYTE_ADDRESS_BUFFER ||
                         m.desc.type == BINDING_RW_TEXTURE2D)
                          ? D3D12_DESCRIPTOR_RANGE_TYPE_UAV
                          : D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    range.NumDescriptors = m.desc.count;
    range.BaseShaderRegister = m.shader_register;
    range.RegisterSpace = m.group;
    range.OffsetInDescriptorsFromTableStart = m.table_offset;
    ranges[m.group * 2 + (m.desc.type == BINDING_SAMPLER)].push_back(range);
  }
  std::vector<D3D12_ROOT_PARAMETER> parameters;
  root_parameters_.assign(layout.groups.size(), {UINT32_MAX, UINT32_MAX});
  for (size_t i = 0; i < ranges.size(); ++i)
    if (!ranges[i].empty()) {
      root_parameters_[i / 2][i % 2] = uint32_t(parameters.size());
      D3D12_ROOT_PARAMETER p{};
      p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
      p.DescriptorTable = {UINT(ranges[i].size()), ranges[i].data()};
      parameters.push_back(p);
    }
  root.NumParameters = UINT(parameters.size());
  root.pParameters = parameters.data();
  ComPtr<ID3DBlob> blob, errors;
  if (!Check(D3D12SerializeRootSignature(&root, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors),
             "Serialize root signature"))
    return false;
  if (!Check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                         IID_PPV_ARGS(&root_signature_)),
             "Create root signature"))
    return false;
  return true;
}
}  // namespace grassland::rhi::backend::d3d12
