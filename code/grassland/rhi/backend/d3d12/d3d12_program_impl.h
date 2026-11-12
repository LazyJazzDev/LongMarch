#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12ProgramImpl final : public Object<rhi::IProgram, ID3D12Program> {
 public:
  static rhi::IProgram *Create(ID3D12Device *device, const ProgramDesc &desc);

  ID3D12PipelineState *NativePipeline() const;

  ~D3D12ProgramImpl() override = default;

  const ProgramDesc &GetDesc() const override;

  ID3D12Device *Owner() const override;

  ID3D12RootSignature *NativeRootSignature() const override;

  uint32_t RootParameter(uint32_t group, bool sampler) const override;

 private:
  D3D12ProgramImpl(ID3D12Device *device, const ProgramDesc &desc);

  ComPtr<ID3D12PipelineState> pipeline_;

  void InitializeProgram(ID3D12Device *device, const ProgramDesc &desc);

  bool ValidateLayout() const;

  ID3D12Device *device_{};

  ProgramDesc desc_{};

  ObjectPtr<IShader> vertex_shader_{};

  ObjectPtr<IShader> fragment_shader_{};

  bool CreateLayout(::ID3D12Device *device, const PipelineLayoutDesc &layout, D3D12_ROOT_SIGNATURE_FLAGS flags);

  std::vector<std::array<uint32_t, 2>> root_parameters_{};
  ComPtr<ID3D12RootSignature> root_signature_{};
};
}  // namespace grassland::rhi::backend::d3d12
