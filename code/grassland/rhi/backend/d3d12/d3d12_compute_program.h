#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_program_base.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"
#include "grassland/rhi/rhi_compute_program.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::d3d12::ID3D12ComputeProgram,
                        grassland::rhi::backend::d3d12::ID3D12ProgramBase,
                        "f0ee363d-3b36-4a53-8e17-b8a0ea11bf0a")

namespace grassland::rhi::backend::d3d12 {
class ID3D12ComputeProgram : public ID3D12ProgramBase {
 public:
  static rhi::IComputeProgram *Create(ID3D12Device *device, const ComputeProgramDesc &desc);
  virtual ID3D12PipelineState *NativePipeline() const = 0;
};

}  // namespace grassland::rhi::backend::d3d12

namespace grassland::rhi::backend::d3d12 {
class D3D12ComputeProgramImpl final : public Object<rhi::IComputeProgram, ID3D12ComputeProgram> {
 public:
  static rhi::IComputeProgram *Create(ID3D12Device *device, const ComputeProgramDesc &desc);

  ID3D12PipelineState *NativePipeline() const;

  ~D3D12ComputeProgramImpl() override = default;

  const ComputeProgramDesc &GetDesc() const override;

  ID3D12Device *Owner() const override;

  ID3D12RootSignature *NativeRootSignature() const override;

  uint32_t RootParameter(uint32_t group, bool sampler) const override;

 private:
  D3D12ComputeProgramImpl(ID3D12Device *device, const ComputeProgramDesc &desc);

  ComPtr<ID3D12PipelineState> pipeline_;

  void InitializeComputeProgram(ID3D12Device *device, const ComputeProgramDesc &desc);

  bool ValidateLayout() const;

  ID3D12Device *device_{};

  ComputeProgramDesc desc_{};

  ObjectPtr<IShader> shader_{};

  bool CreateLayout(::ID3D12Device *device, const PipelineLayoutDesc &layout, D3D12_ROOT_SIGNATURE_FLAGS flags);

  std::vector<std::array<uint32_t, 2>> root_parameters_{};
  ComPtr<ID3D12RootSignature> root_signature_{};
};
}  // namespace grassland::rhi::backend::d3d12
