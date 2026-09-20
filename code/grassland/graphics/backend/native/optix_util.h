#pragma once
#include <optix.h>
#include <optix_stubs.h>

#include "native_memory.h"
#include "native_util.h"

namespace grassland::graphics::backend {

inline void CheckOptix(OptixResult result, const char *operation, const char *log = "") {
  if (result != OPTIX_SUCCESS)
    throw std::runtime_error(std::string("OptiX ") + operation + ": " + optixGetErrorName(result) + "\n" + log);
}

inline CUdeviceptr Pointer(const NativeMemory &memory) {
  return reinterpret_cast<CUdeviceptr>(memory.Data());
}

}  // namespace grassland::graphics::backend
