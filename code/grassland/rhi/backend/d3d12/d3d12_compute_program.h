#pragma once
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_interface_id.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12ComputeProgram)
#include "grassland/rhi/backend/backend_compute_program.h"
#include "grassland/rhi/backend/d3d12/d3d12_program_base.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12ComputeProgram : public IBackendComputeProgram, public ID3D12ProgramBase {
 public:
  virtual void AddRef() noexcept override = 0;
  virtual void Release() noexcept override = 0;
  virtual bool QueryInterface(InterfaceID id, void **output) noexcept override = 0;

  template <typename T>
  T *QueryInterface() noexcept {
    void *result = nullptr;
    return QueryInterface(uuid<T>(), &result) ? static_cast<T *>(result) : nullptr;
  }

 public:
  static ID3D12ComputeProgram *Create(ID3D12Device *device, const ComputeProgramDesc &desc);
  virtual ID3D12PipelineState *NativePipeline() const = 0;
};

;
}  // namespace grassland::rhi::backend::d3d12
