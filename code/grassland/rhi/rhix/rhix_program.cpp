#include "grassland/rhi/rhix/rhix_program.h"

namespace grassland::rhix {
Program::Program() noexcept = default;

Program::Program(rhi::IProgram *object) noexcept : ObjectContainer(object) {
}

rhi::IProgram *Program::Get() const noexcept {
  return static_cast<rhi::IProgram *>(GetObject());
}

const rhi::ProgramDesc &Program::GetDesc() const {
  return Get()->GetDesc();
}

bool Program::Finalize() const {
  return Get()->Finalize();
}

bool Program::IsFinalized() const {
  return Get()->IsFinalized();
}
}  // namespace grassland::rhix
