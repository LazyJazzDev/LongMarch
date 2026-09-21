#pragma once
#include <mutex>

#include "sparkium/backend/cuda/cuda_shader.h"
#include "sparkium/backend/cuda/slang_compiler.h"
#ifdef SPARKIUM_CUDA_ENABLED
#include <cuda.h>
#endif
#ifdef SPARKIUM_OPTIX_ENABLED
#include "sparkium/backend/cuda/optix_launch.h"
#endif
namespace sparkium::backend::cuda {
struct Parameter {
  int slot;
  size_t offset, size;
  bool array;
  SlangTypeKind kind;
  std::string name;
  size_t constant_size{};
};

struct CudaShader::Impl {
  std::string entry;
  std::vector<Parameter> parameters;
  size_t global_size{};
  uint32_t threads[3]{};
#ifdef SPARKIUM_OPTIX_ENABLED
  std::unique_ptr<OptixLaunch> optix_launch;
  OptixDevice *optix_device{};
#endif
#ifdef SPARKIUM_CUDA_ENABLED
  CUmodule module{};
  CUfunction kernel{};
  CUdeviceptr global_device{};
  size_t global_device_size{};
#endif

  ~Impl();
#ifdef SPARKIUM_CUDA_ENABLED
  void CompileCUDA(const std::string &code,
                   const std::string &source,
                   const std::string &compute_entry,
                   bool optix_shader,
                   OptixDevice *optix);
  void DispatchCUDA(const std::vector<uint8_t> &globals, uint32_t x, uint32_t y, uint32_t z);
  void ReleaseCUDA();
#endif
};
}  // namespace sparkium::backend::cuda
