#pragma once
#include "grassland/graphics/backend/webgpu/webgpu_util.h"

namespace grassland::graphics::backend {

// A WGSL module compiled by Slang, offline for the browser.
class WebGPUShader : public Shader {
 public:
  WebGPUShader(WebGPUCore *core, const CompiledShaderBlob &blob);

  std::string EntryPoint() const override {
    return entry_point_;
  }

  const wgpu::ShaderModule &Module() const {
    return module_;
  }

 private:
  std::string entry_point_;
  wgpu::ShaderModule module_;
};

}  // namespace grassland::graphics::backend
