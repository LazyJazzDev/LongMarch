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

bool D3D12Program::Finalize() {
  if (finalized_)
    return true;
  if (!ValidateLayout()) {
    ReportError(-1, "Invalid graphics program layout.");
    return false;
  }
  auto native = static_cast<D3D12Device *>(device_)->NativeDevice();
  std::vector<uint8_t> vertex, fragment;
  auto prefix = ShaderMacroHeader(desc_.pipeline_layout, BACKEND_API_D3D12, desc_.vertex_attributes);
  if (!CompileSlang(desc_.vertex_shader->GetDesc(), prefix, BACKEND_API_D3D12, vertex, &desc_.pipeline_layout))
    return false;
  if (!CompileSlang(desc_.fragment_shader->GetDesc(), prefix, BACKEND_API_D3D12, fragment, &desc_.pipeline_layout))
    return false;
  if (!CreateLayout(native, desc_.pipeline_layout, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT))
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
  state.pRootSignature = NativeRootSignature();
  state.VS = {vertex.data(), vertex.size()};
  state.PS = {fragment.data(), fragment.size()};
  state.InputLayout = {inputs.data(), static_cast<UINT>(inputs.size())};
  state.SampleMask = UINT_MAX;
  state.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
  state.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
  state.RasterizerState.DepthClipEnable = TRUE;
  state.BlendState.IndependentBlendEnable = TRUE;
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
  state.DSVFormat = NativeImageFormat(ds.format);
  state.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
  state.NumRenderTargets = static_cast<UINT>(desc_.color_formats.size());
  for (size_t i = 0; i < desc_.color_formats.size(); ++i) {
    state.RTVFormats[i] = NativeImageFormat(desc_.color_formats[i]);
    const auto value = desc_.blend_states.empty() ? BlendState{} : desc_.blend_states[i];
    auto &blend = state.BlendState.RenderTarget[i];
    blend.BlendEnable = value.blend_enable;
    blend.SrcBlend = NativeBlendFactor(value.src_color);
    blend.DestBlend = NativeBlendFactor(value.dst_color);
    blend.BlendOp = NativeBlendOp(value.color_op);
    blend.SrcBlendAlpha = NativeBlendFactor(value.src_alpha, true);
    blend.DestBlendAlpha = NativeBlendFactor(value.dst_alpha, true);
    blend.BlendOpAlpha = NativeBlendOp(value.alpha_op);
    blend.LogicOp = D3D12_LOGIC_OP_NOOP;
    blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
  }
  state.SampleDesc.Count = 1;
  ComPtr<ID3D12PipelineState> pipeline;
  if (!Check(native->CreateGraphicsPipelineState(&state, IID_PPV_ARGS(&pipeline)), "Create graphics pipeline"))
    return false;
  pipeline_ = std::move(pipeline);
  finalized_ = true;
  return true;
}
}  // namespace grassland::rhi::backend::d3d12
