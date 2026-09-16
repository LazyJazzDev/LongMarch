#pragma once
#include "grassland/graphics/backend/metal/metal_shader.h"
#include "grassland/graphics/backend/metal/metal_util.h"

namespace grassland::graphics::backend {

class MetalComputeProgram : public ComputeProgram {
 public:
  MetalComputeProgram(MetalCore *core, MetalShader *shader) : core(core), shader(shader) {
  }
  void AddResourceBinding(ResourceType type, int count) override;
  void Finalize() override;
  MetalCore *core;
  MetalShader *shader;
  std::vector<MetalBinding> bindings;
  MetalStage stage;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
};
class MetalProgram : public Program {
 public:
  MetalProgram(MetalCore *core, const std::vector<ImageFormat> &colors, ImageFormat depth);
  void AddInputBinding(uint32_t stride, bool per_instance = false) override;
  void AddInputAttribute(uint32_t binding, InputType type, uint32_t offset) override;
  void AddResourceBinding(ResourceType type, int count) override;
  void SetCullMode(CullMode mode) override {
    cull = mode;
  }
  void SetBlendState(int target, const BlendState &state) override;
  void BindShader(Shader *shader, ShaderType type) override;
  void Finalize() override;
  MetalCore *core;
  std::vector<MetalBinding> bindings;
  MetalShader *vertex = nullptr, *fragment = nullptr;
  MetalStage vertex_stage, fragment_stage;
  NS::SharedPtr<MTL::RenderPipelineDescriptor> descriptor;
  NS::SharedPtr<MTL::RenderPipelineState> pipeline;
  NS::SharedPtr<MTL::DepthStencilState> depth_state;
  CullMode cull = CULL_MODE_BACK;
  uint32_t input_bindings = 0, attributes = 0;
};

}  // namespace grassland::graphics::backend
