#pragma once
#include "grassland/rhi/backend/d3d12/d3d12_ray_tracing_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12RayTracingProgramImpl final
    : public Object<rhi::IRayTracingProgram, ID3D12RayTracingProgram, IBackendRayTracingProgram> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IRayTracingProgram,
                            ID3D12RayTracingProgram,
                            IBackendRayTracingProgram,
                            IRayTracingProgram,
                            ID3D12ProgramBase)

  static rhi::IRayTracingProgram *Create(ID3D12Device *device, const RayTracingProgramDesc &desc);

  IShaderBindingTable *CreateShaderBindingTable(const ShaderBindingTableDesc &desc) override;

  const IBackendShaderBindingTable *DefaultShaderBindingTable() const;

  ID3D12StateObject *NativePipeline() const;

  ~D3D12RayTracingProgramImpl() override = default;

  const RayTracingProgramDesc &GetDesc() const override;

  IBackendDevice *Owner() const override;

  ID3D12RootSignature *NativeRootSignature() const override;

  uint32_t RootParameter(uint32_t group, bool sampler) const override;

 private:
  D3D12RayTracingProgramImpl(ID3D12Device *device, const RayTracingProgramDesc &desc);

  ComPtr<ID3D12StateObject> pipeline_;

  IShaderBindingTable *BuildShaderBindingTable(const ShaderBindingTableDesc &desc);

  ObjectPtr<IBackendShaderBindingTable> table_;

  std::array<std::vector<uint8_t>, 4> identifiers_;

  void InitializeRayTracingProgram(ID3D12Device *device, const RayTracingProgramDesc &desc);

  bool ValidateLayout() const;

  bool ValidateShaderBindingTable(const ShaderBindingTableDesc &desc) const;

  ShaderBindingTableDesc DefaultShaderBindingTableDesc() const;

  ID3D12Device *device_{};

  RayTracingProgramDesc desc_{};

  std::vector<ObjectPtr<IShader>> shaders_{};

  bool CreateLayout(::ID3D12Device *device, const PipelineLayoutDesc &layout, D3D12_ROOT_SIGNATURE_FLAGS flags);

  std::vector<std::array<uint32_t, 2>> root_parameters_{};
  ComPtr<ID3D12RootSignature> root_signature_{};
};
}  // namespace grassland::rhi::backend::d3d12
