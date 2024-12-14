#pragma once
#include "grassland/graphics/backend/d3d12/d3d12_core.h"
#include "grassland/graphics/backend/d3d12/d3d12_util.h"

namespace grassland::graphics::backend {
class D3D12Shader : public Shader {
 public:
  D3D12Shader(D3D12Core *core, const void *data, size_t size);
  ~D3D12Shader() override = default;

  ID3DBlob *ShaderBlob() const {
    return shader_blob_.Get();
  }

 private:
  D3D12Core *core_;
  Microsoft::WRL::ComPtr<ID3DBlob> shader_blob_;
};

class D3D12Program : public Program {
 public:
  D3D12Program(D3D12Core *core,
               const std::vector<ImageFormat> &color_formats,
               ImageFormat depth_format);
  void AddInputAttribute(uint32_t binding,
                         InputType type,
                         uint32_t offset) override;
  void AddInputBinding(uint32_t stride, bool input_per_instance) override;
  void BindShader(Shader *shader, ShaderType type) override;
  void Finalize() override;

 private:
  D3D12Core *core_;
  std::vector<std::pair<uint32_t, bool>> input_bindings_;
  std::vector<D3D12_INPUT_ELEMENT_DESC> input_attributes_;
  std::unique_ptr<d3d12::RootSignature> root_signature_;
  D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline_state_desc_;
  std::unique_ptr<d3d12::PipelineState> pipeline_state_;
};

}  // namespace grassland::graphics::backend
