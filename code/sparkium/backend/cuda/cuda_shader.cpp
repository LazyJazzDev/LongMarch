#include <nvrtc.h>

#include "sparkium/backend/common/native_shader_internal.h"
#include "sparkium/backend/common/native_util.h"
#include "sparkium/backend/cuda/cuda_util.h"

namespace sparkium::backend {
namespace {
#ifdef SPARKIUM_NATIVE_CUDA_ENABLED
void CheckNVRTC(nvrtcResult result) {
  if (result != NVRTC_SUCCESS)
    throw std::runtime_error(std::string("native NVRTC: ") + nvrtcGetErrorString(result));
}

struct NVRTCProgramOwner {
  nvrtcProgram program{};

  ~NVRTCProgramOwner() {
    if (program)
      nvrtcDestroyProgram(&program);
  }
};
#endif
}  // namespace

void NativeShader::Impl::CompileCUDA(const std::string &code,
                                     const std::string &source,
                                     const std::string &native_entry,
                                     bool optix_shader,
                                     OptixDevice *optix) {
  NVRTCProgramOwner nvrtc;
  CheckNVRTC(nvrtcCreateProgram(&nvrtc.program, code.c_str(), "sparkium.cu", 0, nullptr, nullptr));
  auto program = nvrtc.program;
  CUdevice device;
  CheckCUDA(cuCtxGetDevice(&device));
  int major, minor;
  CheckCUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  CheckCUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
  std::string arch = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
  std::vector<const char *> options{arch.c_str(), "--std=c++17", "--fmad=false"};
#ifdef SPARKIUM_OPTIX_ENABLED
  if (optix_shader) {
    options.insert(options.end(), {"-DSLANG_CUDA_ENABLE_OPTIX", "-I" LONGMARCH_OPTIX_INCLUDE_DIR,
                                   "-I" SPARKIUM_NATIVE_CUDA_INCLUDE_DIR, "--relocatable-device-code=true"});
  }
#endif
  auto result = nvrtcCompileProgram(program, static_cast<int>(options.size()), options.data());
  size_t log_size = 0;
  CheckNVRTC(nvrtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, '\0');
  CheckNVRTC(nvrtcGetProgramLog(program, log.data()));
  if (result != NVRTC_SUCCESS)
    throw std::runtime_error("NVRTC " + source + ":" + entry + "\n" + log);
  size_t size;
  CheckNVRTC(nvrtcGetPTXSize(program, &size));
  std::string ptx(size, '\0');
  CheckNVRTC(nvrtcGetPTX(program, ptx.data()));
  if (optix_shader) {
#ifdef SPARKIUM_OPTIX_ENABLED
    optix_device = optix;
    optix_launch = std::make_unique<OptixLaunch>(optix, ptx, native_entry, global_size);
#endif
  } else {
    CheckCUDA(cuModuleLoadData(&module, ptx.c_str()));
    CheckCUDA(cuModuleGetFunction(&kernel, module, native_entry.c_str()));
    CheckCUDA(cuModuleGetGlobal(&global_device, &global_device_size, module, "SLANG_globalParams"));
    if (global_device_size < global_size)
      throw std::runtime_error("CUDA global parameter ABI mismatch");
  }
}

void NativeShader::Impl::DispatchCUDA(const std::vector<uint8_t> &globals, uint32_t x, uint32_t y, uint32_t z) {
#ifdef SPARKIUM_OPTIX_ENABLED
  if (optix_launch) {
    if (x > UINT32_MAX / threads[0] || y > UINT32_MAX / threads[1] || z > UINT32_MAX / threads[2])
      throw std::overflow_error("OptiX launch dimensions overflow");
    optix_launch->Dispatch(globals.data(), globals.size(), x * threads[0], y * threads[1], z * threads[2]);
    return;
  }
#endif
  // Slang 2026 uses CUDA module constant memory for global parameters.
  CheckCUDA(cuMemcpyHtoD(global_device, globals.data(), globals.size()));
  CheckCUDA(cuLaunchKernel(kernel, x, y, z, threads[0], threads[1], threads[2], 0, nullptr, nullptr, nullptr));
  // Resource descriptors are launch-local. Keep them alive until completion;
  // errors are attributed to this launch instead of a later image download.
  CheckCUDA(cuCtxSynchronize());
}

void NativeShader::Impl::ReleaseCUDA() {
  if (module)
    cuModuleUnload(module);
}
}  // namespace sparkium::backend
