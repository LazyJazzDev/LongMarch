#include "grassland/graphics/backend/d3d12/d3d12_program.h"

#include <numeric>
#include <stdexcept>

namespace grassland::graphics::backend {

D3D12ProgramBase::D3D12ProgramBase(D3D12Core *core) : core_(core) {
}

void D3D12ProgramBase::AddResourceBindingImpl(ResourceType type, int count) {
  CD3DX12_DESCRIPTOR_RANGE1 range;
  range.Init(ResourceTypeToD3D12DescriptorRangeType(type), count, 0, descriptor_ranges_.size());
  descriptor_ranges_.push_back(range);
}

void D3D12ProgramBase::FinalizeRootSignature() {
  std::vector<CD3DX12_ROOT_PARAMETER1> root_parameters(descriptor_ranges_.size());
  for (size_t i = 0; i < descriptor_ranges_.size(); i++) {
    root_parameters[i].InitAsDescriptorTable(1, &descriptor_ranges_[i], D3D12_SHADER_VISIBILITY_ALL);
  }

  CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC root_signature_desc;
  root_signature_desc.Init_1_1(root_parameters.empty() ? 0 : static_cast<UINT>(root_parameters.size()),
                               root_parameters.empty() ? nullptr : root_parameters.data(), 0, nullptr,
                               D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

  core_->Device()->CreateRootSignature(root_signature_desc, &root_signature_);
}

D3D12Program::D3D12Program(D3D12Core *core, const std::vector<ImageFormat> &color_formats, ImageFormat depth_format)
    : D3D12ProgramBase(core),
      pipeline_state_desc_({}) {
  pipeline_state_desc_.NumRenderTargets = color_formats.size();
  for (size_t i = 0; i < color_formats.size(); i++) {
    pipeline_state_desc_.RTVFormats[i] = ImageFormatToDXGIFormat(color_formats[i]);
  }
  pipeline_state_desc_.DSVFormat = ImageFormatToDXGIFormat(depth_format);
  pipeline_state_desc_.SampleDesc.Count = 1;
  pipeline_state_desc_.SampleDesc.Quality = 0;
  pipeline_state_desc_.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
  pipeline_state_desc_.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
  pipeline_state_desc_.RasterizerState.FrontCounterClockwise = TRUE;
  pipeline_state_desc_.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
  pipeline_state_desc_.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
  if (!IsDepthFormat(depth_format)) {
    pipeline_state_desc_.DepthStencilState.DepthEnable = FALSE;
  }
  pipeline_state_desc_.DepthStencilState.StencilEnable = FALSE;
  pipeline_state_desc_.SampleMask = UINT_MAX;
  pipeline_state_desc_.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
}

void D3D12Program::AddInputAttribute(uint32_t binding, InputType type, uint32_t offset) {
  D3D12_INPUT_ELEMENT_DESC desc{};
  desc.SemanticName = "TEXCOORD";
  desc.SemanticIndex = input_attributes_.size();
  desc.Format = InputTypeToDXGIFormat(type);
  desc.InputSlot = binding;
  desc.AlignedByteOffset = offset;
  desc.InputSlotClass = input_bindings_[binding].second ? D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA
                                                        : D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
  desc.InstanceDataStepRate = input_bindings_[binding].second ? 1 : 0;
  input_attributes_.push_back(desc);
}

void D3D12Program::AddInputBinding(uint32_t stride, bool input_per_instance) {
  input_bindings_.emplace_back(stride, input_per_instance);
}

void D3D12Program::AddResourceBinding(ResourceType type, int count) {
  RecordResourceBinding(type, count);
  AddResourceBindingImpl(type, count);
}

void D3D12Program::SetCullMode(CullMode mode) {
  pipeline_state_desc_.RasterizerState.CullMode = CullModeToD3D12CullMode(mode);
}

void D3D12Program::SetBlendState(int target_id, const BlendState &state) {
  pipeline_state_desc_.BlendState.IndependentBlendEnable = TRUE;
  pipeline_state_desc_.BlendState.RenderTarget[target_id] = BlendStateToD3D12RenderTargetBlendDesc(state);
}

void D3D12Program::BindShader(Shader *shader, ShaderType type) {
  if (!shader)
    throw std::invalid_argument("missing graphics shader");
  shader_stages_[type] = shader;
}

void D3D12Program::Finalize() {
  for (auto [type, shader] : shader_stages_) {
    D3D12Shader *d3d12_shader = dynamic_cast<D3D12Shader *>(ResolveShader(core_, shader));
    if (d3d12_shader) {
      switch (type) {
        case SHADER_TYPE_VERTEX:
          pipeline_state_desc_.VS = d3d12_shader->ShaderModule().Handle();
          break;
        case SHADER_TYPE_PIXEL:
          pipeline_state_desc_.PS = d3d12_shader->ShaderModule().Handle();
          break;
        case SHADER_TYPE_GEOMETRY:
          pipeline_state_desc_.GS = d3d12_shader->ShaderModule().Handle();
          break;
      }
    } else {
      throw std::runtime_error("Invalid shader object, expected D3D12Shader");
    }
  }
  FinalizeRootSignature();

  pipeline_state_desc_.InputLayout.pInputElementDescs = input_attributes_.data();
  pipeline_state_desc_.InputLayout.NumElements = static_cast<UINT>(input_attributes_.size());
  pipeline_state_desc_.pRootSignature = root_signature_->Handle();

  core_->Device()->CreatePipelineState(pipeline_state_desc_, &pipeline_state_);
}

int D3D12Program::NumInputBindings() const {
  return input_bindings_.size();
}

uint32_t D3D12Program::InputBindingStride(uint32_t index) const {
  return input_bindings_[index].first;
}

const D3D12_GRAPHICS_PIPELINE_STATE_DESC *D3D12Program::PipelineStateDesc() const {
  return &pipeline_state_desc_;
}

D3D12ComputeProgram::D3D12ComputeProgram(D3D12Core *core, Shader *compute_shader)
    : D3D12ProgramBase(core),
      compute_shader_(compute_shader) {
}

void D3D12ComputeProgram::AddResourceBinding(ResourceType type, int count) {
  RecordResourceBinding(type, count);
  AddResourceBindingImpl(type, count);
}

void D3D12ComputeProgram::Finalize() {
  auto shader = dynamic_cast<D3D12Shader *>(ResolveShader(core_, compute_shader_));
  if (!shader)
    throw std::invalid_argument("invalid compute shader");
  FinalizeRootSignature();

  D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_desc{};
  pipeline_desc.pRootSignature = root_signature_->Handle();
  pipeline_desc.CS = shader->ShaderModule().Handle();

  core_->Device()->Handle()->CreateComputePipelineState(&pipeline_desc, IID_PPV_ARGS(&pipeline_state_));
}

D3D12RayTracingProgram::D3D12RayTracingProgram(D3D12Core *core,
                                               D3D12Shader *raygen_shader,
                                               D3D12Shader *miss_shader,
                                               D3D12Shader *closest_hit_shader)
    : D3D12RayTracingProgram(core) {
  AddRayGenShader(raygen_shader);
  AddMissShader(miss_shader);
  AddHitGroup({closest_hit_shader, nullptr, nullptr, false});
}

D3D12RayTracingProgram::D3D12RayTracingProgram(D3D12Core *core) : D3D12ProgramBase(core) {
}

void D3D12RayTracingProgram::AddResourceBinding(ResourceType type, int count) {
  RecordResourceBinding(type, count);
  AddResourceBindingImpl(type, count);
}

void D3D12RayTracingProgram::AddRayGenShader(Shader *shader) {
  source_raygen_ = shader;
}

void D3D12RayTracingProgram::AddMissShader(Shader *shader) {
  source_miss_.push_back(shader);
}

void D3D12RayTracingProgram::AddHitGroup(HitGroup group) {
  source_hit_groups_.push_back(group);
}

void D3D12RayTracingProgram::AddCallableShader(Shader *shader) {
  source_callable_.push_back(shader);
}

void D3D12RayTracingProgram::ResolveShaders() {
  auto resolve = [&](Shader *source) -> d3d12::ShaderModule * {
    auto shader = dynamic_cast<D3D12Shader *>(ResolveShader(core_, source));
    if (!shader)
      throw std::invalid_argument("invalid ray tracing shader");
    return &shader->ShaderModule();
  };
  raygen_shader_ = resolve(source_raygen_);
  miss_shaders_.clear();
  hit_groups_.clear();
  callable_shaders_.clear();
  for (auto shader : source_miss_)
    miss_shaders_.push_back(resolve(shader));
  for (auto source : source_hit_groups_) {
    d3d12::HitGroup group{};
    group.closest_hit_shader = source.closest_hit_shader ? resolve(source.closest_hit_shader) : nullptr;
    group.any_hit_shader = source.any_hit_shader ? resolve(source.any_hit_shader) : nullptr;
    group.intersection_shader = source.intersection_shader ? resolve(source.intersection_shader) : nullptr;
    group.procedure = source.procedure;
    hit_groups_.push_back(group);
  }
  for (auto shader : source_callable_)
    callable_shaders_.push_back(resolve(shader));
}

void D3D12RayTracingProgram::Finalize(const std::vector<int32_t> &miss_shader_indices,
                                      const std::vector<int32_t> &hit_group_indices,
                                      const std::vector<int32_t> &callable_shader_indices) {
  ResolveShaders();
  FinalizeRootSignature();

  core_->Device()->CreateRayTracingPipeline(root_signature_.get(), raygen_shader_, miss_shaders_, hit_groups_,
                                            callable_shaders_, &pipeline_);

  core_->Device()->CreateShaderTable(pipeline_.get(), miss_shader_indices, hit_group_indices, callable_shader_indices,
                                     &shader_table_);
}

void D3D12RayTracingProgram::Finalize() {
  std::vector<int32_t> miss_shader_indices(source_miss_.size());
  std::iota(miss_shader_indices.begin(), miss_shader_indices.end(), 0);
  std::vector<int32_t> hit_group_indices(source_hit_groups_.size());
  std::iota(hit_group_indices.begin(), hit_group_indices.end(), 0);
  std::vector<int32_t> callable_shader_indices(source_callable_.size());
  std::iota(callable_shader_indices.begin(), callable_shader_indices.end(), 0);
  Finalize(miss_shader_indices, hit_group_indices, callable_shader_indices);
}

}  // namespace grassland::graphics::backend
