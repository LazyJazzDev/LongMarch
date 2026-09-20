#pragma once
#include <mutex>

#include "sparkium/backend/common/native_shader.h"
#include "sparkium/backend/common/slang_compiler.h"
#ifdef SPARKIUM_NATIVE_CUDA_ENABLED
#include <cuda.h>
#endif
#ifdef SPARKIUM_OPTIX_ENABLED
#include "sparkium/backend/cuda/optix_launch.h"
#endif
namespace sparkium::backend {
// Ordinary exported function: a half-open range of workgroups and grid dimensions.
using HostFunction = void (*)(uint64_t, uint64_t, uint32_t, uint32_t);
using ContextFunction = void (*)(void *, uint64_t, uint64_t, uint32_t, uint32_t);

struct alignas(16) NativeContext {
  uint64_t slots[256]{};
};

struct ConstantField {
  std::string name;
  size_t offset, size;
  void *address{};
};

struct Parameter {
  int slot;
  size_t offset, size;
  bool array;
  SlangTypeKind kind;
  std::string name;
  void *address{};
  std::vector<ConstantField> constants;
  size_t constant_size{};
};

struct NativeShader::Impl {
  bool cuda;
  std::string entry;
  std::vector<Parameter> parameters;
  size_t global_size{};
  uint32_t threads[3]{};
  Slang::ComPtr<ISlangSharedLibrary> library;
#ifdef SPARKIUM_OPTIX_ENABLED
  std::unique_ptr<OptixLaunch> optix_launch;
  OptixDevice *optix_device{};
#endif
  HostFunction host{};
  ContextFunction context_host{};
  bool explicit_context{};
  std::mutex dispatch_mutex;
#ifdef SPARKIUM_NATIVE_CUDA_ENABLED
  CUmodule module{};
  CUfunction kernel{};
  CUdeviceptr global_device{};
  size_t global_device_size{};
#endif

  ~Impl();
  void CompileCPU(const std::filesystem::path &directory,
                  const std::string &source,
                  const std::string &native_entry,
                  const std::string &call_arguments,
                  const std::vector<std::string> &args);
  void DispatchCPU(const std::vector<uint8_t> &globals, uint32_t x, uint32_t y, uint32_t z);
#ifdef SPARKIUM_NATIVE_CUDA_ENABLED
  void CompileCUDA(const std::string &code,
                   const std::string &source,
                   const std::string &native_entry,
                   bool optix_shader,
                   OptixDevice *optix);
  void DispatchCUDA(const std::vector<uint8_t> &globals, uint32_t x, uint32_t y, uint32_t z);
  void ReleaseCUDA();
#endif
};
}  // namespace sparkium::backend
