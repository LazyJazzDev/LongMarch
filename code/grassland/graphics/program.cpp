#include "grassland/graphics/program.h"

#ifdef _WIN64
#include "d3dcompiler.h"
#endif

#if defined(__APPLE__)
#define __EMULATE_UUID
#endif
#include "dxc/dxcapi.h"

namespace CD::graphics {

namespace {
#include "built_in_shaders.inl"
}

#define SAFE_RELEASE(p) \
  if (p) {              \
    p->Release();       \
    p = nullptr;        \
  }

class CustomVFSIncludeHandler : public IDxcIncludeHandler {
 public:
  CustomVFSIncludeHandler(const VirtualFileSystem &vfs,
                          IDxcUtils *dxc_utils,
                          const std::filesystem::path &source_path = "") {
    vfs_ = &vfs;
    dxc_utils_ = dxc_utils;
    source_path_ = source_path;
  }

  // IUnknown methods
  ULONG STDMETHODCALLTYPE AddRef() override {
    return 1;
  }

  ULONG STDMETHODCALLTYPE Release() override {
    return 1;
  }

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void **ppvObject) override {
    if (riid == __uuidof(IDxcIncludeHandler) || riid == __uuidof(IUnknown)) {
      *ppvObject = this;
      AddRef();
      return S_OK;
    }
    *ppvObject = nullptr;
    return E_NOINTERFACE;
  }

  // IDxcIncludeHandler method
  HRESULT STDMETHODCALLTYPE LoadSource(LPCWSTR pFilename,          // The name of the file to include
                                       IDxcBlob **ppIncludeSource  // The file's contents
                                       ) override {
    if (!pFilename || !ppIncludeSource) {
      return E_INVALIDARG;
    }

    // Convert LPCWSTR to std::wstring
    std::string filename = WStringToString(pFilename);

    if (accessed_paths_.count(filename)) {
      HRESULT hr = S_OK;
      IDxcBlobEncoding *source_blob = nullptr;
      hr = dxc_utils_->CreateBlob(nullptr, 0, CP_UTF8, &source_blob);
      if (hr) {
        LogError("Failed to create empty blob for include file: {}", filename);
        return hr;
      }
      source_blob->QueryInterface(IID_PPV_ARGS(ppIncludeSource));
      source_blob->Release();
      return hr;  // Already accessed this file, return success
    } else {
      accessed_paths_.insert(filename);
    }

    std::vector<uint8_t> data;
    if (vfs_->ReadFile(filename, data)) {
#ifndef NDEBUG
      std::wcerr << L"Failed to find include file: " << pFilename << std::endl;
#endif
      return E_FAIL;
    }

    IDxcBlobEncoding *source_blob = nullptr;
    HRESULT hr = S_OK;
    hr = dxc_utils_->CreateBlob(data.data(), static_cast<UINT32>(data.size()), CP_UTF8, &source_blob);
    if (hr) {
      LogError("Failed to create blob from pinned data for include file: {}", filename);
    }
    source_blob->QueryInterface(IID_PPV_ARGS(ppIncludeSource));
    source_blob->Release();
    return hr;
  }

 private:
  ULONG m_ref = 1;
  const VirtualFileSystem *vfs_;
  IDxcUtils *dxc_utils_;
  std::filesystem::path source_path_;
  std::set<std::filesystem::path> accessed_paths_;
};

void RayTracingProgram::AddHitGroup(Shader *closest_hit_shader,
                                    Shader *any_hit_shader,
                                    Shader *intersection_shader,
                                    bool procedure) {
  HitGroup hit_group{
      closest_hit_shader,
      any_hit_shader,
      intersection_shader,
      procedure,
  };
  AddHitGroup(hit_group);
}

CompiledShaderBlob CompileShader(const std::string &source_code,
                                 const std::string &entry_point,
                                 const std::string &target,
                                 const std::vector<std::string> &args) {
  VirtualFileSystem vfs;
  vfs.WriteFile("shader.hlsl", source_code);
  return CompileShader(vfs, "shader.hlsl", entry_point, target, args);
}

CompiledShaderBlob CompileShader(const VirtualFileSystem &vfs,
                                 const std::string &source_file,
                                 const std::string &entry_point,
                                 const std::string &target,
                                 const std::vector<std::string> &args) {
  HRESULT hr;

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
  IDxcCompilerArgs *dxc_args{nullptr};
  IDxcResult *result{nullptr};
  IDxcBlob *source_blob{nullptr};

  DeferredProcess dxc_release([&]() {
    SAFE_RELEASE(dxc_compiler);
    SAFE_RELEASE(dxc_utils);
    SAFE_RELEASE(dxc_args);
    SAFE_RELEASE(result);
    SAFE_RELEASE(source_blob);
  });

  DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxc_compiler));
  DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxc_utils));

  CustomVFSIncludeHandler include_handler{vfs, dxc_utils, source_file};

  std::vector<std::wstring> wargs;
  for (const auto &arg : args) {
    wargs.push_back(StringToWString(arg));
  }

  if (target[0] == 'l' && target[1] == 'i' && target[2] == 'b') {
    wargs.push_back(StringToWString("-exports"));
    wargs.push_back(StringToWString(entry_point));
  }

#if !defined(NDEBUG)
  wargs.emplace_back(L"-Zi");
  wargs.emplace_back(L"-DDEBUG_SHADER");
#endif

  std::vector<LPCWSTR> warg_ptrs;
  for (const auto &arg : wargs) {
    warg_ptrs.push_back(arg.c_str());
  }

  dxc_utils->BuildArguments(StringToWString(source_file).c_str(), StringToWString(entry_point).c_str(),
                            StringToWString(target).c_str(), warg_ptrs.data(), wargs.size(), nullptr, 0, &dxc_args);

  include_handler.LoadSource(StringToWString(source_file).c_str(), &source_blob);

  DxcBuffer dxc_buffer;
  dxc_buffer.Ptr = source_blob->GetBufferPointer();
  dxc_buffer.Size = source_blob->GetBufferSize();
  dxc_buffer.Encoding = CP_UTF8;

  hr = dxc_compiler->Compile(&dxc_buffer, dxc_args->GetArguments(), dxc_args->GetCount(), &include_handler,
                             IID_PPV_ARGS(&result));
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

}  // namespace CD::graphics
