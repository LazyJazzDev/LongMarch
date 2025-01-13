#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_core.h"
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {
class D3D12Shader : public Shader {
 public:
  D3D12Shader(D3D12Core *core, const CompiledShaderBlob &shader_blob);
  ~D3D12Shader() override = default;

  const d3d12::ShaderModule &ShaderModule() const {
    return shader_module_;
  }

 private:
  D3D12Core *core_;
  d3d12::ShaderModule shader_module_;
};

class D3D12Program : public Program {
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

  d3d12::RootSignature *RootSignature() const {
    return root_signature_.get();
  }

  CD3DX12_DESCRIPTOR_RANGE1 *DescriptorRange(int index) {
    return &descriptor_ranges_[index];
  }

 private:
  D3D12Core *core_;
  std::vector<std::pair<uint32_t, bool>> input_bindings_;
  std::vector<D3D12_INPUT_ELEMENT_DESC> input_attributes_;
  std::vector<CD3DX12_DESCRIPTOR_RANGE1> descriptor_ranges_;
  std::unique_ptr<d3d12::RootSignature> root_signature_;
  D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline_state_desc_;
  std::unique_ptr<d3d12::PipelineState> pipeline_state_;
};

}  // namespace grassland::graphics::backend
