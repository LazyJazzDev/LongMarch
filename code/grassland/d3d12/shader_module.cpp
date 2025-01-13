#include "grassland/d3d12/shader_module.h"

#include <dxcapi.h>

#include "grassland/graphics/program.h"

namespace grassland::d3d12 {
ShaderModule::ShaderModule(const CompiledShaderBlob &shader_blob)
    : shader_code_(shader_blob.data), entry_point_(shader_blob.entry_point) {
}

ComPtr<ID3DBlob> CompileShaderLegacy(const std::string &source_code,
                                     const std::string &entry_point,
                                     const std::string &target) {
#if defined(_DEBUG)
  // Enable better shader debugging with the graphics debugging tools.
  UINT compile_flags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
  UINT compile_flags = 0;
#endif

  ComPtr<ID3DBlob> shader_blob;
  ComPtr<ID3DBlob> error_blob;
  auto hr = D3DCompile(source_code.c_str(), source_code.size(), nullptr, nullptr, nullptr, entry_point.c_str(),
                       target.c_str(), compile_flags, 0, &shader_blob, &error_blob);
  if (FAILED(hr)) {
    if (error_blob) {
      LogError("Failed to compile shader: {}", static_cast<char *>(error_blob->GetBufferPointer()));
    } else {
      LogError("Failed to compile shader.");
    }
    return nullptr;
  }
  return shader_blob;
}

CompiledShaderBlob CompileShader(const std::string &source_code,
                                 const std::string &entry_point,
                                 const std::string &target) {
  return graphics::CompileShader(source_code, entry_point, target);
}

}  // namespace grassland::d3d12
