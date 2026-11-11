#include "grassland/rhi/backend/d3d12/d3d12_program.h"

#include <unordered_set>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_program_impl.h"

namespace grassland::rhi::backend::d3d12 {
D3D12ProgramImpl::D3D12ProgramImpl(ID3D12Device *device, const ProgramDesc &desc) {
  InitializeProgram(device, desc);
}

ID3D12PipelineState *D3D12ProgramImpl::NativePipeline() const {
  return pipeline_.Get();
}

rhi::IProgram *D3D12ProgramImpl::Create(ID3D12Device *owner_device, const ProgramDesc &desc) {
  auto object = std::unique_ptr<D3D12ProgramImpl>(new D3D12ProgramImpl(owner_device, desc));
  if (!object->ValidateLayout()) {
    ReportError(-1, "Invalid graphics program layout.");
    return nullptr;
  }
  auto native = owner_device->NativeDevice();
  std::vector<uint8_t> vertex, fragment;
  auto prefix = ShaderMacroHeader(object->desc_.pipeline_layout, BACKEND_API_D3D12, object->desc_.vertex_attributes);
  if (!CompileSlang(object->desc_.vertex_shader, prefix, BACKEND_API_D3D12, vertex, &object->desc_.pipeline_layout))
    return nullptr;
  if (!CompileSlang(object->desc_.fragment_shader, prefix, BACKEND_API_D3D12, fragment, &object->desc_.pipeline_layout))
    return nullptr;
  if (!object->CreateLayout(native, object->desc_.pipeline_layout,
                            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT))
    return nullptr;
  std::vector<D3D12_INPUT_ELEMENT_DESC> inputs;
  const DXGI_FORMAT formats[] = {
      DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT,
      DXGI_FORMAT_R32_SINT,  DXGI_FORMAT_R32G32_SINT,  DXGI_FORMAT_R32G32B32_SINT,  DXGI_FORMAT_R32G32B32A32_SINT,
      DXGI_FORMAT_R32_UINT,  DXGI_FORMAT_R32G32_UINT,  DXGI_FORMAT_R32G32B32_UINT,  DXGI_FORMAT_R32G32B32A32_UINT};
  for (const auto &a : object->desc_.vertex_attributes) {
    auto instance = object->desc_.vertex_bindings[a.binding].per_instance;
    inputs.push_back(
        {"TEXCOORD", a.location, formats[a.type], a.binding, a.offset,
         instance ? D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA : D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
         instance ? 1u : 0u});
  }
  D3D12_GRAPHICS_PIPELINE_STATE_DESC state{};
  state.pRootSignature = object->NativeRootSignature();
  state.VS = {vertex.data(), vertex.size()};
  state.PS = {fragment.data(), fragment.size()};
  state.InputLayout = {inputs.data(), static_cast<UINT>(inputs.size())};
  state.SampleMask = UINT_MAX;
  state.RasterizerState.FillMode =
      desc.rasterizer.fill_mode == FILL_MODE_WIREFRAME ? D3D12_FILL_MODE_WIREFRAME : D3D12_FILL_MODE_SOLID;
  const D3D12_CULL_MODE cull_modes[] = {D3D12_CULL_MODE_NONE, D3D12_CULL_MODE_FRONT, D3D12_CULL_MODE_BACK};
  state.RasterizerState.CullMode = cull_modes[desc.rasterizer.cull_mode];
  state.RasterizerState.FrontCounterClockwise = desc.rasterizer.front_face == FRONT_FACE_COUNTER_CLOCKWISE;
  state.RasterizerState.DepthClipEnable = TRUE;
  state.BlendState.IndependentBlendEnable = TRUE;
  const auto &ds = object->desc_.depth_stencil;
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
  state.PrimitiveTopologyType =
      desc.topology == PRIMITIVE_TOPOLOGY_POINT_LIST ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT
      : (desc.topology == PRIMITIVE_TOPOLOGY_LINE_LIST || desc.topology == PRIMITIVE_TOPOLOGY_LINE_STRIP)
          ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE
          : D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
  state.NumRenderTargets = static_cast<UINT>(object->desc_.color_formats.size());
  for (size_t i = 0; i < object->desc_.color_formats.size(); ++i) {
    state.RTVFormats[i] = NativeImageFormat(object->desc_.color_formats[i]);
    const auto value = object->desc_.blend_states.empty() ? BlendState{} : object->desc_.blend_states[i];
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
    return nullptr;
  object->pipeline_ = std::move(pipeline);
  return object.release();
}

rhi::IProgram *ID3D12Program::Create(ID3D12Device *device, const ProgramDesc &desc) {
  return D3D12ProgramImpl::Create(device, desc);
}

void D3D12ProgramImpl::InitializeProgram(IBackendDevice *device, const ProgramDesc &desc) {
  device_ = device;
  desc_ = desc;
  vertex_shader_ = ObjectPtr<IShader>::Retain(desc.vertex_shader);
  fragment_shader_ = ObjectPtr<IShader>::Retain(desc.fragment_shader);
}

const ProgramDesc &D3D12ProgramImpl::GetDesc() const {
  return desc_;
}

IBackendDevice *D3D12ProgramImpl::Owner() const {
  return device_;
}

bool D3D12ProgramImpl::ValidateLayout() const {
  if (desc_.topology < PRIMITIVE_TOPOLOGY_TRIANGLE_LIST || desc_.topology > PRIMITIVE_TOPOLOGY_POINT_LIST ||
      desc_.rasterizer.cull_mode < CULL_MODE_NONE || desc_.rasterizer.cull_mode > CULL_MODE_BACK ||
      desc_.rasterizer.front_face < FRONT_FACE_COUNTER_CLOCKWISE ||
      desc_.rasterizer.front_face > FRONT_FACE_CLOCKWISE || desc_.rasterizer.fill_mode < FILL_MODE_SOLID ||
      desc_.rasterizer.fill_mode > FILL_MODE_WIREFRAME)
    return false;
  if (desc_.pipeline_layout.groups.size() > 8)
    return false;
  for (const auto &group : desc_.pipeline_layout.groups)
    if (!ValidateBindingLayout(group))
      return false;
  if (!desc_.vertex_shader || !desc_.fragment_shader)
    return false;
  if (desc_.vertex_shader->GetStage() != SHADER_STAGE_VERTEX)
    return false;
  if (desc_.fragment_shader->GetStage() != SHADER_STAGE_FRAGMENT)
    return false;
  if (desc_.color_formats.size() > 8 || (desc_.color_formats.empty() && !IsDepthFormat(desc_.depth_stencil.format)))
    return false;
  for (auto format : desc_.color_formats)
    if (!IsColorFormat(format))
      return false;
  if (!desc_.blend_states.empty() && desc_.blend_states.size() != desc_.color_formats.size())
    return false;
  for (size_t i = 0; i < desc_.blend_states.size(); ++i) {
    if (desc_.blend_states[i].blend_enable &&
        (IsSignedIntegerFormat(desc_.color_formats[i]) || IsUnsignedIntegerFormat(desc_.color_formats[i])))
      return false;
  }
  for (const auto &blend : desc_.blend_states) {
    for (auto factor : {blend.src_color, blend.dst_color, blend.src_alpha, blend.dst_alpha})
      if (factor < BLEND_FACTOR_ZERO || factor > BLEND_FACTOR_ONE_MINUS_DST_ALPHA)
        return false;
    for (auto op : {blend.color_op, blend.alpha_op})
      if (op < BLEND_OP_ADD || op > BLEND_OP_MAX)
        return false;
  }
  const auto &depth = desc_.depth_stencil;
  if (depth.format != IMAGE_FORMAT_UNKNOWN && !IsDepthFormat(depth.format))
    return false;
  if ((depth.depth_test || depth.depth_write || depth.stencil_test) && !IsDepthFormat(depth.format))
    return false;
  if (depth.stencil_test && !HasStencil(depth.format))
    return false;
  if (depth.depth_compare < COMPARE_NEVER || depth.depth_compare > COMPARE_ALWAYS)
    return false;
  for (auto face : {depth.front, depth.back}) {
    if (face.compare < COMPARE_NEVER || face.compare > COMPARE_ALWAYS)
      return false;
    for (auto op : {face.fail_op, face.depth_fail_op, face.pass_op})
      if (op < STENCIL_KEEP || op > STENCIL_DECREMENT_WRAP)
        return false;
  }
  std::unordered_set<uint32_t> locations;
  for (const auto &attribute : desc_.vertex_attributes) {
    if (attribute.binding >= desc_.vertex_bindings.size())
      return false;
    auto size = InputTypeSize(attribute.type);
    auto stride = desc_.vertex_bindings[attribute.binding].stride;
    if (!size || attribute.offset > stride || size > stride - attribute.offset)
      return false;
    if (!locations.insert(attribute.location).second)
      return false;
  }
  return true;
}

ID3D12RootSignature *D3D12ProgramImpl::NativeRootSignature() const {
  return root_signature_.Get();
}

uint32_t D3D12ProgramImpl::RootParameter(uint32_t group, bool sampler) const {
  return root_parameters_[group][sampler];
}

bool D3D12ProgramImpl::CreateLayout(::ID3D12Device *device,
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

LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::ID3D12Program, "2919454c-7ff9-4ce3-b919-40901e038398")
