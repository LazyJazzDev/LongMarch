#pragma once
#include "grassland/rhi/rhi_program.h"
#include "grassland/rhi/rhix/rhix_object_container.h"

namespace grassland::rhix {
class Program : public ObjectContainer {
 public:
  Program() noexcept;
  Program(rhi::IProgram *object) noexcept;
  rhi::IProgram *Get() const noexcept;
  bool Finalize() const;
  bool IsFinalized() const;
  const rhi::ProgramDesc &GetDesc() const;
};
}  // namespace grassland::rhix
