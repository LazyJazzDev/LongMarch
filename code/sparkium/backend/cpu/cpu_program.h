#pragma once
#include "grassland/graphics/program.h"
#include "sparkium/backend/cpu/cpu_shader.h"

namespace sparkium::backend::cpu {
using namespace grassland;
using namespace grassland::graphics;

class CpuProgram final : public ComputeProgram {
 public:
  explicit CpuProgram(CpuShader *shader);

  void AddResourceBinding(ResourceType type, int count) override;

  void Finalize() override;

  CpuShader *shader;
  std::vector<std::pair<ResourceType, int>> resources;
};

}  // namespace sparkium::backend::cpu
