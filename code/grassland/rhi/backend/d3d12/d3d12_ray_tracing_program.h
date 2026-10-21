#pragma once
#include "grassland/rhi/backend/backend_ray_tracing_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_allocated_buffer.h"
#include "grassland/rhi/backend/d3d12/d3d12_program_base.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Device;

class D3D12RayTracingProgram final : public RayTracingProgram, public D3D12ProgramBase {
 public:
  D3D12RayTracingProgram(D3D12Device *device, const RayTracingProgramDesc &desc);
  bool Finalize() override;
  ID3D12StateObject *NativePipeline() const;
  D3D12_DISPATCH_RAYS_DESC DispatchDesc() const;

 private:
  ComPtr<ID3D12StateObject> pipeline_;
  ObjectPtr<D3D12AllocatedBuffer> table_;
  D3D12_DISPATCH_RAYS_DESC dispatch_{};
};
}  // namespace grassland::rhi::backend::d3d12
