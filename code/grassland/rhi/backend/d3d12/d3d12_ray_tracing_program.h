#pragma once
#include "grassland/rhi/backend/backend_ray_tracing_program.h"
#include "grassland/rhi/backend/backend_shader_binding_table.h"
#include "grassland/rhi/backend/d3d12/d3d12_allocated_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_program_base.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

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
