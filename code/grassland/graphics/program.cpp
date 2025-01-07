#include "grassland/graphics/program.h"

#ifdef _WIN32
#include "d3dcompiler.h"
#endif
#include "directx-dxc/dxcapi.h"

namespace grassland::graphics {
#define SAFE_RELEASE(p) \
  if (p) {              \
    p->Release();       \
    p = nullptr;        \
  }

CompiledShaderBlob CompileShader(const std::string &source_code, const std::string &entry_point, const std::string &target, const std::vector<std::string> &args) {
  CompiledShaderBlob shader_blob;
  shader_blob.entry_point = entry_point;

#if defined(_DEBUG)
  // Enable better shader debugging with the graphics debugging tools.
  UINT compile_flags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
  UINT compile_flags = 0;
#endif
  IDxcCompiler3 *dxc_compiler{nullptr};
  IDxcUtils *dxc_utils{nullptr};
  IDxcBlobEncoding *source_blob{nullptr};
  IDxcCompilerArgs *dxc_args{nullptr};
  IDxcResult *result{nullptr};

  DeferredProcess dxc_release([&]() {
    SAFE_RELEASE(dxc_compiler);
    SAFE_RELEASE(dxc_utils);
    SAFE_RELEASE(source_blob);
    SAFE_RELEASE(dxc_args);
    SAFE_RELEASE(result);
  });

  DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxc_compiler));
  DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxc_utils));

  HRESULT hr;
  hr = dxc_utils->CreateBlobFromPinned(source_code.c_str(), source_code.size(), CP_UTF8, &source_blob);
  if (FAILED(hr)) {
    LogError("Failed to create blob with encoding from pinned.");
  }

  std::vector<std::wstring> wargs;
  for (const auto &arg : args) {
    wargs.push_back(StringToWString(arg));
  }
  std::vector<LPCWSTR> warg_ptrs;
  for (const auto &arg : wargs) {
    warg_ptrs.push_back(arg.c_str());
  }

  dxc_utils->BuildArguments(nullptr, StringToWString(entry_point).c_str(), StringToWString(target).c_str(), warg_ptrs.data(), wargs.size(), nullptr, 0, &dxc_args);

  DxcBuffer dxc_buffer;
  dxc_buffer.Ptr = source_blob->GetBufferPointer();
  dxc_buffer.Size = source_blob->GetBufferSize();
  dxc_buffer.Encoding = CP_UTF8;

  hr = dxc_compiler->Compile(&dxc_buffer, dxc_args->GetArguments(), dxc_args->GetCount(), nullptr, IID_PPV_ARGS(&result));
#ifndef IDxcBlobUtf16
#define IDxcBlobUtf16 IDxcBlobWide
#endif
  if (result->HasOutput(DXC_OUT_ERRORS)) {
    IDxcBlobUtf8 *error_blob{nullptr};
    IDxcBlobUtf16 *output_name{nullptr};
    result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&error_blob), &output_name);

    if (error_blob->GetStringPointer() && error_blob->GetStringLength()) {
      LogInfo("{}", error_blob->GetStringPointer());
    }
    SAFE_RELEASE(error_blob);
    SAFE_RELEASE(output_name);
  }

  if (result->HasOutput(DXC_OUT_OBJECT)) {
    IDxcBlob *object_blob{nullptr};
    IDxcBlobUtf16 *output_name{nullptr};
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&object_blob), &output_name);

    shader_blob.data.resize(object_blob->GetBufferSize());
    std::memcpy(shader_blob.data.data(), object_blob->GetBufferPointer(), object_blob->GetBufferSize());
    SAFE_RELEASE(object_blob);
    SAFE_RELEASE(output_name);
  } else {
    LogError("Failed to compile shader.");
  }

  return shader_blob;
}
}  // namespace grassland::graphics
