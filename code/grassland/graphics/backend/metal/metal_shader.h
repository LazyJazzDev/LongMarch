#pragma once
#include <map>

#include "grassland/graphics/backend/metal/metal_util.h"

namespace grassland::graphics::backend {

class MetalShader : public Shader {
 public:
  explicit MetalShader(const CompiledShaderBlob &blob) : blob(blob) {
  }

  std::string EntryPoint() const override {
    return blob.entry_point;
  }

  CompiledShaderBlob blob;
};

struct MetalStage {
  NS::SharedPtr<MTL::Function> function;
  std::map<int, NS::SharedPtr<MTL::ArgumentEncoder>> arguments;
  MTL::Size threads{1, 1, 1};
};

MetalStage CompileMetalStage(MetalCore *core, MetalShader *shader, const std::vector<MetalBinding> &bindings);

}  // namespace grassland::graphics::backend
