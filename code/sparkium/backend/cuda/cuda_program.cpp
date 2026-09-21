#include "sparkium/backend/cuda/cuda_program.h"

namespace sparkium::backend::cuda {

CudaProgram::CudaProgram(CudaShader *shader) : shader(shader) {
}

void CudaProgram::AddResourceBinding(ResourceType type, int count) {
  resources.emplace_back(type, count);
}

void CudaProgram::Finalize() {
}

}  // namespace sparkium::backend::cuda
