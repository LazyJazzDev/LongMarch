#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_compute_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12ComputeProgramImpl final
    : public Object<rhi::IComputeProgram, ID3D12ComputeProgram, IBackendComputeProgram> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IComputeProgram,
                            ID3D12ComputeProgram,
                            IBackendComputeProgram,
                            IComputeProgram,
                            ID3D12ProgramBase)

  static rhi::IComputeProgram *Create(ID3D12Device *device, const ComputeProgramDesc &desc);

  ID3D12PipelineState *NativePipeline() const;

  ~D3D12ComputeProgramImpl() override = default;

  const ComputeProgramDesc &GetDesc() const override;

  IBackendDevice *Owner() const override;

  ID3D12RootSignature *NativeRootSignature() const override;

  uint32_t RootParameter(uint32_t group, bool sampler) const override;

 private:
  D3D12ComputeProgramImpl(ID3D12Device *device, const ComputeProgramDesc &desc);

  ComPtr<ID3D12PipelineState> pipeline_;

  void InitializeComputeProgram(IBackendDevice *device, const ComputeProgramDesc &desc);

  bool ValidateLayout() const;

  IBackendDevice *device_{};

  ComputeProgramDesc desc_{};

  ObjectPtr<IShader> shader_{};

  bool CreateLayout(::ID3D12Device *device, const PipelineLayoutDesc &layout, D3D12_ROOT_SIGNATURE_FLAGS flags);

  std::vector<std::array<uint32_t, 2>> root_parameters_{};
  ComPtr<ID3D12RootSignature> root_signature_{};
};
}  // namespace grassland::rhi::backend::d3d12
