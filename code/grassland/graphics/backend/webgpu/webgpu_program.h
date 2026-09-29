#pragma once
#include <map>
#include <vector>

#include "grassland/graphics/backend/webgpu/webgpu_util.h"

namespace grassland::graphics::backend {

// Each AddResourceBinding call is one bind group; Slang emits @group(slot) @binding(0).
std::vector<wgpu::BindGroupLayout> CreateBindGroupLayouts(WebGPUCore *core,
                                                          const std::vector<WebGPUBinding> &bindings,
                                                          wgpu::ShaderStage visibility);

class WebGPUComputeProgram : public ComputeProgram {
 public:
  WebGPUComputeProgram(WebGPUCore *core, WebGPUShader *shader) : core_(core), shader_(shader) {
  }

  void AddResourceBinding(ResourceType type, int count) override;
  void Finalize() override;

  const std::vector<WebGPUBinding> &Bindings() const {
    return bindings_;
  }

  const std::vector<wgpu::BindGroupLayout> &Layouts() const {
    return layouts_;
  }

  const wgpu::ComputePipeline &Pipeline() const {
    return pipeline_;
  }

 private:
  WebGPUCore *core_;
  WebGPUShader *shader_;
  std::vector<WebGPUBinding> bindings_;
  std::vector<wgpu::BindGroupLayout> layouts_;
  wgpu::ComputePipeline pipeline_;
};

class WebGPUProgram : public Program {
 public:
  WebGPUProgram(WebGPUCore *core, const std::vector<ImageFormat> &colors, ImageFormat depth);
  void AddInputBinding(uint32_t stride, bool per_instance = false) override;
  void AddInputAttribute(uint32_t binding, InputType type, uint32_t offset) override;
  void AddResourceBinding(ResourceType type, int count) override;

  void SetCullMode(CullMode mode) override {
    cull_ = mode;
  }

  void SetBlendState(int target, const BlendState &state) override;
  void BindShader(Shader *shader, ShaderType type) override;
  void Finalize() override;

  // WebGPU bakes the topology into the pipeline; the engine sets it per draw.
  const wgpu::RenderPipeline &Pipeline(PrimitiveTopology topology);

  const std::vector<WebGPUBinding> &Bindings() const {
    return bindings_;
  }

  const std::vector<wgpu::BindGroupLayout> &Layouts() const {
    return layouts_;
  }

 private:
  struct VertexBinding {
    uint32_t stride;
    bool per_instance;
    std::vector<wgpu::VertexAttribute> attributes;
  };

  WebGPUCore *core_;
  std::vector<ImageFormat> colors_;
  ImageFormat depth_;
  std::vector<VertexBinding> vertex_bindings_;
  uint32_t attributes_ = 0;
  std::vector<WebGPUBinding> bindings_;
  std::map<int, BlendState> blends_;
  WebGPUShader *vertex_ = nullptr, *fragment_ = nullptr;
  CullMode cull_ = CULL_MODE_BACK;
  std::vector<wgpu::BindGroupLayout> layouts_;
  wgpu::PipelineLayout layout_;
  std::map<PrimitiveTopology, wgpu::RenderPipeline> pipelines_;
};

}  // namespace grassland::graphics::backend
