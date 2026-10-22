#include "grassland/rhi/backend/d3d12/d3d12_program_base.h"

#include "grassland/rhi/backend/backend_binding_group.h"

namespace grassland::rhi::backend::d3d12 {
D3D12ProgramBase::D3D12ProgramBase() = default;
D3D12ProgramBase::~D3D12ProgramBase() = default;

ID3D12RootSignature *D3D12ProgramBase::NativeRootSignature() const {
  return root_signature_.Get();
}

uint32_t D3D12ProgramBase::RootParameter(uint32_t group, bool sampler) const {
  return root_parameters_[group][sampler];
}

bool D3D12ProgramBase::CreateLayout(ID3D12Device *device,
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
