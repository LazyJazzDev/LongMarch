#pragma once
#include "grassland/rhi/backend/backend_ray_tracing_program_state.h"
#include "grassland/rhi/backend/d3d12/d3d12_program_base_state.h"
#include "grassland/rhi/backend/d3d12/d3d12_ray_tracing_program.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12RayTracingProgramImpl final : public Object<ID3D12RayTracingProgram> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(IRayTracingProgram,
                            ID3D12RayTracingProgram,
                            RayTracingProgram,
                            IRayTracingProgram,
                            D3D12ProgramBase,
                            IObject)
  static ID3D12RayTracingProgram *Create(ID3D12Device *device, const RayTracingProgramDesc &desc);
  IShaderBindingTable *CreateShaderBindingTable(const ShaderBindingTableDesc &desc) override;
  const ShaderBindingTable *DefaultShaderBindingTable() const;
  ID3D12StateObject *NativePipeline() const;

 private:
  D3D12RayTracingProgramImpl(ID3D12Device *device, const RayTracingProgramDesc &desc);
  ComPtr<ID3D12StateObject> pipeline_;
  ShaderBindingTable *BuildShaderBindingTable(const ShaderBindingTableDesc &desc);
  ObjectPtr<ShaderBindingTable> table_;
  std::array<std::vector<uint8_t>, 4> identifiers_;

 public:
  ~D3D12RayTracingProgramImpl() override {
  }

 private:
  RayTracingProgramState state_RayTracingProgram_;

  RayTracingProgramState &StateRayTracingProgram() override {
    return state_RayTracingProgram_;
  }

  const RayTracingProgramState &StateRayTracingProgram() const override {
    return state_RayTracingProgram_;
  }

 private:
  D3D12ProgramBaseState state_D3D12ProgramBase_;

  D3D12ProgramBaseState &StateD3D12ProgramBase() override {
    return state_D3D12ProgramBase_;
  }

  const D3D12ProgramBaseState &StateD3D12ProgramBase() const override {
    return state_D3D12ProgramBase_;
  }
};
}  // namespace grassland::rhi::backend::d3d12
