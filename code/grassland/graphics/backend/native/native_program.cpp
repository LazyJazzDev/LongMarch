#include "native_program.h"

namespace grassland::graphics::backend {

NativeProgram::NativeProgram(NativeShader *shader) : shader(shader) {
}

void NativeProgram::AddResourceBinding(ResourceType type, int count) {
  resources.emplace_back(type, count);
}

void NativeProgram::Finalize() {
}

}  // namespace grassland::graphics::backend
