#include "grassland/rhi/rhix/rhix_program.h"

namespace grassland::rhix {
Program::Program() noexcept = default;

Program::Program(rhi::IProgram *object) noexcept : ObjectContainer(object) {
}

rhi::IProgram *Program::Get() const noexcept {
  return rhi::QueryInterfacePtr<rhi::IProgram>(GetObject()).Get();
}

const rhi::ProgramDesc &Program::GetDesc() const {
  return Get()->GetDesc();
}

}  // namespace grassland::rhix
