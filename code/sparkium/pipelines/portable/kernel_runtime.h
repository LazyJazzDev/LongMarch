#pragma once

#include <functional>
#include <memory>
#include <string>

#include "sparkium/pipelines/portable/hlsl_compat.h"
#include "sparkium/pipelines/portable/scene_bake.h"

namespace sparkium::portable {

// Compiles kernel C++ source (hlsl_compat.h + generated code) at runtime and
// returns the PortableRenderPixel entry point. Results are cached by source
// hash. The compilation uses the same toolchain that built the project.
class KernelLibrary {
 public:
  virtual ~KernelLibrary() = default;
  virtual RenderPixelFn Entry() const = 0;
  // Address of the module's sparkium_portable::g_ctx_ptr (host kernels only).
  virtual void *ContextSlot() const {
    return nullptr;
  }
};

// Compiles `source` (or returns a cached build). `cache_directory` stores
// objects between runs; `label` is used in diagnostics.
std::shared_ptr<KernelLibrary> CompileKernelHost(const std::string &source,
                                                 const std::string &cache_directory,
                                                 const std::string &label);

#ifdef SPARKIUM_PORTABLE_CUDA
// CUDA path: loads the kernel through the CUDA driver after nvcc compilation.
std::shared_ptr<KernelLibrary> CompileKernelCuda(const std::string &source,
                                                 const std::string &cache_directory,
                                                 const std::string &label);
#endif

// SHA-256 of `text`, hex-encoded (used for cache keys).
std::string HashString(const std::string &text);

}  // namespace sparkium::portable
