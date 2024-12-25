#include "grassland/d3d12/shader_module.h"

#include <dxcapi.h>

namespace grassland::d3d12 {
ShaderModule::ShaderModule(const std::vector<uint8_t> &shader_code)
    : shader_code_(shader_code) {
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
  auto hr = D3DCompile(source_code.c_str(), source_code.size(), nullptr,
                       nullptr, nullptr, entry_point.c_str(), target.c_str(),
                       compile_flags, 0, &shader_blob, &error_blob);
  if (FAILED(hr)) {
    if (error_blob) {
      LogError("Failed to compile shader: {}",
               static_cast<char *>(error_blob->GetBufferPointer()));
    } else {
      LogError("Failed to compile shader.");
    }
    return nullptr;
  }
  return shader_blob;
}

ComPtr<ID3DBlob> CompileShader(const std::string &source_code,
                               const std::string &entry_point,
                               const std::string &target) {
#if defined(_DEBUG)
  // Enable better shader debugging with the graphics debugging tools.
  UINT compile_flags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
  UINT compile_flags = 0;
#endif
  ComPtr<IDxcCompiler3> dxc_compiler;
  ComPtr<IDxcUtils> dxc_utils;
  ComPtr<IDxcBlobEncoding> source_blob;
  DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxc_compiler));
  DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxc_utils));

  HRESULT hr;
  hr = dxc_utils->CreateBlobFromPinned(source_code.c_str(), source_code.size(),
                                       CP_UTF8, &source_blob);
  if (FAILED(hr)) {
    LogError("Failed to create blob with encoding from pinned.");
  }

  ComPtr<IDxcCompilerArgs> dxc_args;
  dxc_utils->BuildArguments(nullptr, StringToWString(entry_point).c_str(),
                            StringToWString(target).c_str(), nullptr, 0,
                            nullptr, 0, &dxc_args);

  DxcBuffer dxc_buffer;
  dxc_buffer.Ptr = source_blob->GetBufferPointer();
  dxc_buffer.Size = source_blob->GetBufferSize();
  dxc_buffer.Encoding = CP_UTF8;

  ComPtr<IDxcResult> result;
  hr = dxc_compiler->Compile(&dxc_buffer, dxc_args->GetArguments(),
                             dxc_args->GetCount(), nullptr,
                             IID_PPV_ARGS(&result));

  if (result->HasOutput(DXC_OUT_ERRORS)) {
    ComPtr<IDxcBlobUtf8> error_blob;
    ComPtr<IDxcBlobUtf16> output_name;
    result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&error_blob), &output_name);

    if (error_blob->GetStringPointer() && error_blob->GetStringLength()) {
      LogInfo("Error: {}", error_blob->GetStringPointer());
    }
  }

  ComPtr<ID3DBlob> shader_blob;
  if (result->HasOutput(DXC_OUT_OBJECT)) {
    ComPtr<IDxcBlob> object_blob;
    ComPtr<IDxcBlobUtf16> output_name;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&object_blob), &output_name);

    D3DCreateBlob(object_blob->GetBufferSize(), &shader_blob);
    std::memcpy(shader_blob->GetBufferPointer(),
                object_blob->GetBufferPointer(), object_blob->GetBufferSize());
  } else {
    LogError("Failed to compile shader.");
  }

  return shader_blob;
}

}  // namespace grassland::d3d12
