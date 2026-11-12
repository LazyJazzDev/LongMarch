#pragma once
#include "grassland/rhi/backend/backend_shader_binding_table.h"
#include "grassland/rhi/backend/d3d12/d3d12_allocated_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_program_base.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::d3d12::ID3D12RayTracingProgram,
                        grassland::rhi::backend::d3d12::ID3D12ProgramBase,
                        "5d1da804-a5b3-4469-a380-9d7a44cee916")

namespace grassland::rhi::backend::d3d12 {
class ID3D12RayTracingProgram : public ID3D12ProgramBase {
 public:
  static rhi::IRayTracingProgram *Create(ID3D12Device *device, const RayTracingProgramDesc &desc);
  virtual const IBackendShaderBindingTable *DefaultShaderBindingTable() const = 0;
  virtual ID3D12StateObject *NativePipeline() const = 0;
};

}  // namespace grassland::rhi::backend::d3d12

namespace grassland::rhi::backend::d3d12 {
class D3D12RayTracingProgramImpl final : public Object<rhi::IRayTracingProgram, ID3D12RayTracingProgram> {
 public:
  static rhi::IRayTracingProgram *Create(ID3D12Device *device, const RayTracingProgramDesc &desc);

  IShaderBindingTable *CreateShaderBindingTable(const ShaderBindingTableDesc &desc) override;

  const IBackendShaderBindingTable *DefaultShaderBindingTable() const;

  ID3D12StateObject *NativePipeline() const;

  ~D3D12RayTracingProgramImpl() override = default;

  const RayTracingProgramDesc &GetDesc() const override;

  ID3D12Device *Owner() const override;

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
