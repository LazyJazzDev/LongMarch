#pragma once
#include "grassland/rhi/backend/backend_ray_tracing_program.h"
#include "grassland/rhi/backend/backend_shader_binding_table.h"
#include "grassland/rhi/backend/d3d12/d3d12_allocated_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_program_base.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Device;

class D3D12RayTracingProgram final : public RayTracingProgram, public D3D12ProgramBase {
 public:
  D3D12RayTracingProgram(D3D12Device *device, const RayTracingProgramDesc &desc);
  bool Finalize() override;
  IShaderBindingTable *CreateShaderBindingTable(const ShaderBindingTableDesc &desc) override;
  const ShaderBindingTable *DefaultShaderBindingTable() const;
  ID3D12StateObject *NativePipeline() const;

 private:
  ComPtr<ID3D12StateObject> pipeline_;
  ShaderBindingTable *BuildShaderBindingTable(const ShaderBindingTableDesc &desc);
  ObjectPtr<ShaderBindingTable> table_;
  std::array<std::vector<uint8_t>, 4> identifiers_;
};
}  // namespace grassland::rhi::backend::d3d12
