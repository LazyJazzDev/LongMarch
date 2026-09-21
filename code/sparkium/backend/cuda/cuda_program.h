#pragma once
#include "grassland/graphics/program.h"
#include "sparkium/backend/cuda/cuda_shader.h"

namespace sparkium::backend::cuda {
using namespace grassland;
using namespace grassland::graphics;

class CudaProgram final : public ComputeProgram {
 public:
  explicit CudaProgram(CudaShader *shader);

  void AddResourceBinding(ResourceType type, int count) override;

  void Finalize() override;

  CudaShader *shader;
  std::vector<std::pair<ResourceType, int>> resources;
};

}  // namespace sparkium::backend::cuda
