#include "sparkium/backend/cpu/cpu_program.h"

namespace sparkium::backend::cpu {

CpuProgram::CpuProgram(CpuShader *shader) : shader(shader) {
}

void CpuProgram::AddResourceBinding(ResourceType type, int count) {
  resources.emplace_back(type, count);
}

void CpuProgram::Finalize() {
}

}  // namespace sparkium::backend::cpu
