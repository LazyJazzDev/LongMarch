#include "grassland/rhi/backend/d3d12/d3d12_program.h"

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
D3D12Program::D3D12Program(D3D12Device *device, const ProgramDesc &desc) : Program(device, desc) {
}

ID3D12PipelineState *D3D12Program::NativePipeline() const {
  return pipeline_.Get();
}

ID3D12RootSignature *D3D12Program::NativeRootSignature() const {
  return root_signature_.Get();
}

uint32_t D3D12Program::RootParameter(uint32_t group, bool sampler) const {
  return root_parameters_[group][sampler];
}

bool D3D12Program::Finalize() {
  if (finalized_)
    return true;
  try {
    if (!ValidateLayout()) {
      ReportError(-1, "Invalid graphics program layout.");
      return false;
    }
    auto native = static_cast<D3D12Device *>(device_)->NativeDevice();
    std::vector<uint8_t> vertex, fragment;
    auto prefix = ShaderMacroHeader(desc_, BACKEND_API_D3D12);
    if (!CompileSlang(desc_.vertex_shader->GetDesc(), prefix, BACKEND_API_D3D12, vertex, &desc_.pipeline_layout))
      return false;
    if (!CompileSlang(desc_.fragment_shader->GetDesc(), prefix, BACKEND_API_D3D12, fragment, &desc_.pipeline_layout))
      return false;
    D3D12_ROOT_SIGNATURE_DESC root{};
    root.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    const auto mappings = MapBindings(desc_.pipeline_layout, BACKEND_API_D3D12);
    std::vector<std::vector<D3D12_DESCRIPTOR_RANGE>> ranges(desc_.pipeline_layout.groups.size() * 2);
    for (auto m : mappings) {
      D3D12_DESCRIPTOR_RANGE range{};
      range.RangeType = m.desc.type == BINDING_SAMPLER          ? D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER
                        : m.desc.type == BINDING_UNIFORM_BUFFER ? D3D12_DESCRIPTOR_RANGE_TYPE_CBV
                        : (m.desc.type == BINDING_RW_STORAGE_BUFFER || m.desc.type == BINDING_RW_TEXTURE2D)
                            ? D3D12_DESCRIPTOR_RANGE_TYPE_UAV
                            : D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
      range.NumDescriptors = m.desc.count;
      range.BaseShaderRegister = m.shader_register;
      range.RegisterSpace = m.group;
      range.OffsetInDescriptorsFromTableStart = m.table_offset;
      ranges[m.group * 2 + (m.desc.type == BINDING_SAMPLER)].push_back(range);
    }
    std::vector<D3D12_ROOT_PARAMETER> parameters;
    root_parameters_.assign(desc_.pipeline_layout.groups.size(), {UINT32_MAX, UINT32_MAX});
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
    ComPtr<ID3D12RootSignature> signature;
    if (!Check(
            native->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&signature)),
            "Create root signature"))
      return false;
    std::vector<D3D12_INPUT_ELEMENT_DESC> inputs;
    const DXGI_FORMAT formats[] = {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT,
                                   DXGI_FORMAT_R32G32B32A32_FLOAT};
    for (const auto &a : desc_.vertex_attributes) {
      auto instance = desc_.vertex_bindings[a.binding].per_instance;
      inputs.push_back(
          {"TEXCOORD", a.location, formats[a.format], a.binding, a.offset,
           instance ? D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA : D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
           instance ? 1u : 0u});
    }
    D3D12_GRAPHICS_PIPELINE_STATE_DESC state{};
    state.pRootSignature = signature.Get();
    state.VS = {vertex.data(), vertex.size()};
    state.PS = {fragment.data(), fragment.size()};
    state.InputLayout = {inputs.data(), static_cast<UINT>(inputs.size())};
    state.SampleMask = UINT_MAX;
    state.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    state.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    state.RasterizerState.DepthClipEnable = TRUE;
    auto &blend = state.BlendState.RenderTarget[0];
    blend.SrcBlend = D3D12_BLEND_ONE;
    blend.DestBlend = D3D12_BLEND_ZERO;
    blend.BlendOp = D3D12_BLEND_OP_ADD;
    blend.SrcBlendAlpha = D3D12_BLEND_ONE;
    blend.DestBlendAlpha = D3D12_BLEND_ZERO;
    blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    blend.LogicOp = D3D12_LOGIC_OP_NOOP;
    blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    const auto &ds = desc_.depth_stencil;
    state.DepthStencilState.DepthEnable = ds.depth_test;
    state.DepthStencilState.DepthWriteMask = ds.depth_write ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    state.DepthStencilState.DepthFunc = static_cast<D3D12_COMPARISON_FUNC>(ds.depth_compare + 1);
    state.DepthStencilState.StencilEnable = ds.stencil_test;
    state.DepthStencilState.StencilReadMask = ds.stencil_read_mask;
    state.DepthStencilState.StencilWriteMask = ds.stencil_write_mask;
    auto face = [](const StencilFaceDesc &value) {
      D3D12_DEPTH_STENCILOP_DESC result{};
      result.StencilFailOp = static_cast<D3D12_STENCIL_OP>(value.fail_op + 1);
      result.StencilDepthFailOp = static_cast<D3D12_STENCIL_OP>(value.depth_fail_op + 1);
      result.StencilPassOp = static_cast<D3D12_STENCIL_OP>(value.pass_op + 1);
      result.StencilFunc = static_cast<D3D12_COMPARISON_FUNC>(value.compare + 1);
      return result;
    };
    state.DepthStencilState.FrontFace = face(ds.front);
    state.DepthStencilState.BackFace = face(ds.back);
    state.DSVFormat = ds.format == IMAGE_FORMAT_D32_FLOAT           ? DXGI_FORMAT_D32_FLOAT
                      : ds.format == IMAGE_FORMAT_D24_UNORM_S8_UINT ? DXGI_FORMAT_D24_UNORM_S8_UINT
                                                                    : DXGI_FORMAT_UNKNOWN;
    state.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    state.NumRenderTargets = static_cast<UINT>(desc_.color_formats.size());
    for (size_t i = 0; i < desc_.color_formats.size(); ++i) {
      state.RTVFormats[i] =
          desc_.color_formats[i] == IMAGE_FORMAT_RGBA8_UNORM ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_B8G8R8A8_UNORM;
      state.BlendState.RenderTarget[i] = blend;
    }
    state.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> pipeline;
    if (!Check(native->CreateGraphicsPipelineState(&state, IID_PPV_ARGS(&pipeline)), "Create graphics pipeline"))
      return false;
    root_signature_ = std::move(signature);
    pipeline_ = std::move(pipeline);
    finalized_ = true;
    return true;
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Program allocation failed.");
    return false;
  }
}
}  // namespace grassland::rhi::backend::d3d12
