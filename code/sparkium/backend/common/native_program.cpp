#include "sparkium/backend/common/native_program.h"

namespace sparkium::backend {

NativeProgram::NativeProgram(NativeShader *shader) : shader(shader) {
}

void NativeProgram::AddResourceBinding(ResourceType type, int count) {
  resources.emplace_back(type, count);
}

void NativeProgram::Finalize() {
}

}  // namespace sparkium::backend
