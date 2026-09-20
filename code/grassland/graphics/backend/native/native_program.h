#pragma once
#include "grassland/graphics/program.h"
#include "native_shader.h"

namespace grassland::graphics::backend {

class NativeProgram final : public ComputeProgram {
 public:
  explicit NativeProgram(NativeShader *shader);

  void AddResourceBinding(ResourceType type, int count) override;

  void Finalize() override;

  NativeShader *shader;
  std::vector<std::pair<ResourceType, int>> resources;
};

}  // namespace grassland::graphics::backend
