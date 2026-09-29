#include "grassland/graphics/backend/webgpu/webgpu_shader.h"

#include <sstream>
#include <string>

#include "grassland/graphics/backend/webgpu/webgpu_core.h"

namespace grassland::graphics::backend {

namespace {
// Slang copies @interpolate onto structs that a shader also uses as ordinary
// values (such as PSInput passed to helper functions). WGSL allows it only on
// entry-point inputs and outputs, so drop it from fields without @location.
std::string WithoutPlainInterpolation(const std::string &code) {
  std::istringstream input(code);
  std::string result, line;
  while (std::getline(input, line)) {
    const auto start = line.find("@interpolate(");
    if (start != std::string::npos && line.find("@location") == std::string::npos) {
      const auto end = line.find(')', start);
      if (end != std::string::npos)
        line.erase(start, end + 1 - start);
    }
    result += line;
    result += '\n';
  }
  return result;
}
}  // namespace

WebGPUShader::WebGPUShader(WebGPUCore *core, const CompiledShaderBlob &blob) : entry_point_(blob.entry_point) {
  const std::string code = WithoutPlainInterpolation(std::string(blob.data.begin(), blob.data.end()));
  wgpu::ShaderSourceWGSL source{};
  source.code = {code.data(), code.size()};
  wgpu::ShaderModuleDescriptor descriptor{};
  descriptor.nextInChain = &source;
  module_ = core->Device().CreateShaderModule(&descriptor);
}

}  // namespace grassland::graphics::backend
