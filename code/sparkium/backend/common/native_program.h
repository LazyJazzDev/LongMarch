#pragma once
#include "grassland/graphics/program.h"
#include "sparkium/backend/common/native_shader.h"

namespace sparkium::backend {
using namespace grassland;
using namespace grassland::graphics;

class NativeProgram final : public ComputeProgram {
 public:
  explicit NativeProgram(NativeShader *shader);

  void AddResourceBinding(ResourceType type, int count) override;

  void Finalize() override;

  NativeShader *shader;
  std::vector<std::pair<ResourceType, int>> resources;
};

}  // namespace sparkium::backend
