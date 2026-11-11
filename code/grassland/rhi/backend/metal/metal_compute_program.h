#pragma once
#include "grassland/rhi/backend/backend_compute_program.h"
#include "grassland/rhi/backend/metal/metal_program_base.h"
#include "grassland/rhi/backend/metal/metal_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::metal::IMetalComputeProgram)

namespace grassland::rhi::backend::metal {
class IMetalComputeProgram : public IBackendComputeProgram, public IMetalProgramBase {
 public:
  virtual void AddRef() noexcept override = 0;
  virtual void Release() noexcept override = 0;
  virtual bool QueryInterface(InterfaceID id, void **output) noexcept override = 0;

  template <typename T>
  T *QueryInterface() noexcept {
    void *result = nullptr;
    return QueryInterface(uuid<T>(), &result) ? static_cast<T *>(result) : nullptr;
  }

  static IMetalComputeProgram *Create(IMetalDevice *device, const ComputeProgramDesc &desc);
  virtual MTL::ComputePipelineState *NativePipeline() const = 0;
  virtual MTL::Size ThreadGroupSize() const = 0;
};

}  // namespace grassland::rhi::backend::metal
