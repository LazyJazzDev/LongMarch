#pragma once
#include "grassland/rhi/rhi_compute_program.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
class ComputeProgram : public ObjectContainer {
 public:
  ComputeProgram() noexcept;
  ComputeProgram(rhi::IComputeProgram *object) noexcept;
  rhi::IComputeProgram *Get() const noexcept;
  bool Finalize() const;
  bool IsFinalized() const;
  const rhi::ComputeProgramDesc &GetDesc() const;
};
}  // namespace grassland::rhix
