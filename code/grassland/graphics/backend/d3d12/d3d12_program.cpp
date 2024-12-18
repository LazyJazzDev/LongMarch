#include "grassland/graphics/backend/d3d12/d3d12_program.h"

namespace grassland::graphics::backend {

D3D12Shader::D3D12Shader(D3D12Core *core, const void *data, size_t size)
    : core_(core) {
  D3DCreateBlob(size, &shader_blob_);
  memcpy(shader_blob_->GetBufferPointer(), data, size);
}

D3D12Program::D3D12Program(D3D12Core *core,
                           const std::vector<ImageFormat> &color_formats,
                           ImageFormat depth_format)
    : core_(core), pipeline_state_desc_({}) {
  pipeline_state_desc_.NumRenderTargets = color_formats.size();
  for (size_t i = 0; i < color_formats.size(); i++) {
    pipeline_state_desc_.RTVFormats[i] =
        ImageFormatToDXGIFormat(color_formats[i]);
  }
  pipeline_state_desc_.DSVFormat = ImageFormatToDXGIFormat(depth_format);
  pipeline_state_desc_.SampleDesc.Count = 1;
  pipeline_state_desc_.SampleDesc.Quality = 0;
  pipeline_state_desc_.PrimitiveTopologyType =
      D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
  pipeline_state_desc_.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
  pipeline_state_desc_.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
  pipeline_state_desc_.DepthStencilState =
      CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
  if (!IsDepthFormat(depth_format)) {
    pipeline_state_desc_.DepthStencilState.DepthEnable = FALSE;
  }
  pipeline_state_desc_.DepthStencilState.StencilEnable = FALSE;
  pipeline_state_desc_.SampleMask = UINT_MAX;
  pipeline_state_desc_.IBStripCutValue =
      D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
}

void D3D12Program::AddInputAttribute(uint32_t binding,
                                     InputType type,
                                     uint32_t offset) {
  D3D12_INPUT_ELEMENT_DESC desc{};
  desc.SemanticName = "TEXCOORD";
  desc.SemanticIndex = input_attributes_.size();
  desc.Format = InputTypeToDXGIFormat(type);
  desc.InputSlot = binding;
  desc.AlignedByteOffset = offset;
  desc.InputSlotClass = input_bindings_[binding].second
                            ? D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA
                            : D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
  desc.InstanceDataStepRate = input_bindings_[binding].second ? 1 : 0;
  input_attributes_.push_back(desc);
}

void D3D12Program::AddInputBinding(uint32_t stride, bool input_per_instance) {
  input_bindings_.emplace_back(stride, input_per_instance);
}

void D3D12Program::BindShader(Shader *shader, ShaderType type) {
  D3D12Shader *d3d12_shader = dynamic_cast<D3D12Shader *>(shader);
  if (d3d12_shader) {
    switch (type) {
      case SHADER_TYPE_VERTEX:
        pipeline_state_desc_.VS =
            CD3DX12_SHADER_BYTECODE(d3d12_shader->ShaderBlob());
        break;
      case SHADER_TYPE_FRAGMENT:
        pipeline_state_desc_.PS =
            CD3DX12_SHADER_BYTECODE(d3d12_shader->ShaderBlob());
        break;
    }
  } else {
    throw std::runtime_error("Invalid shader object, expected D3D12Shader");
  }
}

void D3D12Program::Finalize() {
  CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC root_signature_desc;
  root_signature_desc.Init_1_1(
      0, nullptr, 0, nullptr,
      D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

  core_->Device()->CreateRootSignature(root_signature_desc, &root_signature_);

  pipeline_state_desc_.InputLayout.pInputElementDescs =
      input_attributes_.data();
  pipeline_state_desc_.InputLayout.NumElements =
      static_cast<UINT>(input_attributes_.size());
  pipeline_state_desc_.pRootSignature = root_signature_->Handle();

  core_->Device()->CreatePipelineState(pipeline_state_desc_, &pipeline_state_);
}

}  // namespace grassland::graphics::backend
