#include "grassland/rhi/rhix/rhix_compute_program.h"

namespace grassland::rhix {
ComputeProgram::ComputeProgram() noexcept = default;

ComputeProgram::ComputeProgram(rhi::IComputeProgram *object) noexcept : ObjectContainer(object) {
}

rhi::IComputeProgram *ComputeProgram::Get() const noexcept {
  return rhi::QueryInterfacePtr<rhi::IComputeProgram>(GetObject()).Get();
}

const rhi::ComputeProgramDesc &ComputeProgram::GetDesc() const {
  return Get()->GetDesc();
}

}  // namespace grassland::rhix
