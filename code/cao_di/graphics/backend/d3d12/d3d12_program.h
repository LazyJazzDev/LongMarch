#pragma once
#include "cao_di/graphics/backend/d3d12/d3d12_core.h"
#include "cao_di/graphics/backend/d3d12/d3d12_util.h"

namespace CD::graphics::backend {
class D3D12Shader : public Shader {
 public:
  D3D12Shader(D3D12Core *core, const CompiledShaderBlob &shader_blob);
  ~D3D12Shader() override = default;

  d3d12::ShaderModule &ShaderModule() {
    return shader_module_;
  }

  const d3d12::ShaderModule &ShaderModule() const {
    return shader_module_;
  }

 private:
  D3D12Core *core_;
  d3d12::ShaderModule shader_module_;
};

class D3D12ProgramBase {
 public:
  D3D12ProgramBase(D3D12Core *core);
  virtual ~D3D12ProgramBase() = default;

  void AddResourceBindingImpl(ResourceType type, int count);
  void FinalizeRootSignature();

  d3d12::RootSignature *RootSignature() const {
    return root_signature_.get();
  }

  CD3DX12_DESCRIPTOR_RANGE1 *DescriptorRange(int index) {
    return &descriptor_ranges_[index];
  }

 protected:
  D3D12Core *core_;
  std::vector<CD3DX12_DESCRIPTOR_RANGE1> descriptor_ranges_;
  std::unique_ptr<d3d12::RootSignature> root_signature_;
};

class D3D12Program : public Program, public D3D12ProgramBase {
 public:
  D3D12Program(D3D12Core *core, const std::vector<ImageFormat> &color_formats, ImageFormat depth_format);
  void AddInputAttribute(uint32_t binding, InputType type, uint32_t offset) override;
  void AddInputBinding(uint32_t stride, bool input_per_instance) override;
  void AddResourceBinding(ResourceType type, int count) override;
  void SetCullMode(CullMode mode) override;
  void SetBlendState(int target_id, const BlendState &state) override;
  void BindShader(Shader *shader, ShaderType type) override;
  void Finalize() override;

  int NumInputBindings() const;
  uint32_t InputBindingStride(uint32_t index) const;
  const D3D12_GRAPHICS_PIPELINE_STATE_DESC *PipelineStateDesc() const;

  d3d12::PipelineState *PipelineState() const {
    return pipeline_state_.get();
  }

 private:
  std::vector<std::pair<uint32_t, bool>> input_bindings_;
  std::vector<D3D12_INPUT_ELEMENT_DESC> input_attributes_;
  D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline_state_desc_;
  std::unique_ptr<d3d12::PipelineState> pipeline_state_;
};

class D3D12ComputeProgram : public ComputeProgram, public D3D12ProgramBase {
 public:
  D3D12ComputeProgram(D3D12Core *core, D3D12Shader *compute_shader);
  ~D3D12ComputeProgram() override = default;
  void AddResourceBinding(ResourceType type, int count) override;
  void Finalize() override;

  d3d12::ComPtr<ID3D12PipelineState> PipelineState() const {
    return pipeline_state_;
  }

 private:
  D3D12Shader *compute_shader_;
  d3d12::ComPtr<ID3D12PipelineState> pipeline_state_;
};

class D3D12RayTracingProgram : public RayTracingProgram, public D3D12ProgramBase {
 public:
  D3D12RayTracingProgram(D3D12Core *core,
                         D3D12Shader *raygen_shader,
                         D3D12Shader *miss_shader,
                         D3D12Shader *closest_hit_shader);
  D3D12RayTracingProgram(D3D12Core *core);
  ~D3D12RayTracingProgram() override = default;
  void AddResourceBinding(ResourceType type, int count) override;

  void AddRayGenShader(Shader *ray_gen_shader) override;
  void AddMissShader(Shader *miss_shader) override;
  void AddHitGroup(HitGroup hit_group) override;
  void AddCallableShader(Shader *callable_shader) override;

  void Finalize(const std::vector<int32_t> &miss_shader_indices,
                const std::vector<int32_t> &hit_group_indices,
                const std::vector<int32_t> &callable_shader_indices) override;

  d3d12::RayTracingPipeline *PipelineState() const {
    return pipeline_.get();
  }

  d3d12::ShaderTable *ShaderTable() const {
    return shader_table_.get();
  }

 private:
  d3d12::ShaderModule *raygen_shader_;
  std::vector<d3d12::ShaderModule *> miss_shaders_;
  std::vector<d3d12::HitGroup> hit_groups_;
  std::vector<d3d12::ShaderModule *> callable_shaders_;
  std::unique_ptr<d3d12::RayTracingPipeline> pipeline_;
  std::unique_ptr<d3d12::ShaderTable> shader_table_;
};

}  // namespace CD::graphics::backend
