#include "native_core.h"

#include <iostream>

#include "native_buffer.h"
#include "native_command_context.h"
#include "native_image.h"
#include "native_program.h"
#include "native_sampler.h"
#include "native_util.h"
#ifdef LONGMARCH_OPTIX_ENABLED
#include "native_acceleration_structure.h"
#include "optix_device.h"
#endif

namespace grassland::graphics::backend {

NativeCore::NativeCore(BackendAPI api, const Settings &settings) : Core(settings), api_(api) {
}

NativeCore::~NativeCore() {
#ifdef LONGMARCH_OPTIX_ENABLED
  optix_.reset();
#endif
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
  if (cuda_context_) {
    cuCtxSynchronize();
    cuDevicePrimaryCtxRelease(device_index_);
  }
#endif
}

int NativeCore::GetPhysicalDeviceProperties(PhysicalDeviceProperties *p) {
  if (api_ == BACKEND_API_CPU) {
    if (p) {
      p[0].name = "Native CPU (Slang LLVM JIT)";
      p[0].score = 1;
      p[0].ray_tracing_support = false;
      p[0].geometry_shader_support = false;
    }
    return 1;
  }
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
  CheckCUDA(cuInit(0));
  int count = 0;
  CheckCUDA(cuDeviceGetCount(&count));
  if (p)
    for (int i = 0; i < count; ++i) {
      char name[256];
      CheckCUDA(cuDeviceGetName(name, sizeof(name), i));
      p[i].name = name;
      p[i].score = 1;
      p[i].ray_tracing_support = false;
      p[i].geometry_shader_support = false;
      p[i].cuda_device_index = i;
#ifdef LONGMARCH_OPTIX_ENABLED
      // Probe the driver as well as the GPU, so hardware-only auto selection
      // agrees with the capability exposed after logical-device initialization.
      CUcontext previous{}, probe{};
      CheckCUDA(cuCtxGetCurrent(&previous));
      if (cuDevicePrimaryCtxRetain(&probe, i) == CUDA_SUCCESS) {
        try {
          CheckCUDA(cuCtxSetCurrent(probe));
          OptixDevice device(probe, false);
          p[i].ray_tracing_support = true;
        } catch (const std::exception &) {
          p[i].ray_tracing_support = false;
        }
        CheckCUDA(cuCtxSetCurrent(previous));
        CheckCUDA(cuDevicePrimaryCtxRelease(i));
      }
#endif
    }
  return count;
#else
  return 0;
#endif
}

int NativeCore::InitializeLogicalDevice(int index) {
  int count = GetPhysicalDeviceProperties();
  if (index < 0 || index >= count)
    return -1;
  std::vector<PhysicalDeviceProperties> properties(count);
  GetPhysicalDeviceProperties(properties.data());
  device_name_ = properties[index].name;
  device_index_ = index;
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
  if (UsesCUDA()) {
    CUcontext context;
    CheckCUDA(cuDevicePrimaryCtxRetain(&context, index));
    cuda_context_ = context;
    cuda_device_ = index;
    CheckCUDA(cuCtxSetCurrent(context));
#ifdef LONGMARCH_OPTIX_ENABLED
    try {
      optix_ = std::make_unique<OptixDevice>(context, DebugEnabled());
      ray_tracing_support_ = true;
    } catch (const std::exception &error) {
      optix_.reset();
      ray_tracing_support_ = false;
      std::cerr << "CUDA OptiX ray tracing unavailable: " << error.what() << '\n';
    }
#endif
  }
#endif
  return 0;
}

void NativeCore::WaitGPU() {
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
  if (UsesCUDA())
    CheckCUDA(cuCtxSynchronize());
#endif
}

int NativeCore::CreateBuffer(size_t s, BufferType t, double_ptr<Buffer> p) {
  p.construct<NativeBuffer>(UsesCUDA(), s, t);
  return 0;
}

int NativeCore::CreateImage(int w, int h, ImageFormat f, double_ptr<Image> p) {
  p.construct<NativeImage>(UsesCUDA(), w, h, f);
  return 0;
}

int NativeCore::CreateSampler(const SamplerInfo &i, double_ptr<Sampler> p) {
  p.construct<NativeSampler>(i);
  return 0;
}

int NativeCore::CreateShader(const std::string &s, const std::string &e, const std::string &t, double_ptr<Shader> p) {
  VirtualFileSystem v;
  v.WriteFile("input.hlsl", s);
  return CreateShader(v, "input.hlsl", e, t, {}, p);
}

int NativeCore::CreateShader(const VirtualFileSystem &v,
                             const std::string &s,
                             const std::string &e,
                             const std::string &t,
                             double_ptr<Shader> p) {
  return CreateShader(v, s, e, t, {}, p);
}

int NativeCore::CreateShader(const VirtualFileSystem &v,
                             const std::string &s,
                             const std::string &e,
                             const std::string &t,
                             const std::vector<std::string> &a,
                             double_ptr<Shader> p) {
  if (t.rfind("cs_", 0) != 0)
    NativeUnsupported();
  OptixDevice *optix = nullptr;
#ifdef LONGMARCH_OPTIX_ENABLED
  optix = optix_.get();
#endif
  p.construct<NativeShader>(UsesCUDA(), v, s, e, a, optix);
  return 0;
}

int NativeCore::CreateComputeProgram(Shader *s, double_ptr<ComputeProgram> p) {
  auto *n = dynamic_cast<NativeShader *>(s);
  if (!n)
    throw std::runtime_error("foreign native shader");
  p.construct<NativeProgram>(n);
  return 0;
}

int NativeCore::CreateCommandContext(double_ptr<CommandContext> p) {
  p.construct<NativeCommandContext>(this);
  return 0;
}

int NativeCore::SubmitCommandContext(CommandContext *p) {
  auto *n = dynamic_cast<NativeCommandContext *>(p);
  if (!n)
    throw std::runtime_error("foreign native command context");
  for (auto &f : n->commands)
    f();
  WaitGPU();
  for (auto &f : n->GetPostExecutionCallbacks())
    f();
  return 0;
}

int NativeCore::CreateWindowObject(int, int, const std::string &, bool, bool, double_ptr<Window>) {
  NativeUnsupported();
}

int NativeCore::CreateProgram(const std::vector<ImageFormat> &, ImageFormat, double_ptr<Program>) {
  NativeUnsupported();
}

int NativeCore::CreateBottomLevelAccelerationStructure(BufferRange,
                                                       uint32_t,
                                                       uint32_t,
                                                       RayTracingGeometryFlag,
                                                       double_ptr<AccelerationStructure>) {
  NativeUnsupported();
}

int NativeCore::CreateBottomLevelAccelerationStructure(BufferRange vertices,
                                                       BufferRange indices,
                                                       uint32_t vertex_count,
                                                       uint32_t stride,
                                                       uint32_t primitive_count,
                                                       RayTracingGeometryFlag flags,
                                                       double_ptr<AccelerationStructure> output) {
#ifdef LONGMARCH_OPTIX_ENABLED
  if (optix_) {
    output.construct<OptixAccelerationStructure>(optix_.get(), vertices, indices, vertex_count, stride, primitive_count,
                                                 flags);
    return 0;
  }
#endif
  NativeUnsupported();
}

int NativeCore::CreateBottomLevelAccelerationStructure(Buffer *vertices,
                                                       Buffer *indices,
                                                       uint32_t stride,
                                                       double_ptr<AccelerationStructure> output) {
  if (!vertices || !indices || !stride || vertices->Size() / stride > UINT32_MAX || indices->Size() / 12 > UINT32_MAX)
    throw std::invalid_argument("invalid native triangle geometry");
  return CreateBottomLevelAccelerationStructure(vertices->Range(), indices->Range(), vertices->Size() / stride, stride,
                                                indices->Size() / 12, RAYTRACING_GEOMETRY_FLAG_NONE, output);
}

int NativeCore::CreateTopLevelAccelerationStructure(const std::vector<RayTracingInstance> &instances,
                                                    double_ptr<AccelerationStructure> output) {
#ifdef LONGMARCH_OPTIX_ENABLED
  if (optix_) {
    output.construct<OptixAccelerationStructure>(optix_.get(), instances);
    return 0;
  }
#endif
  NativeUnsupported();
}

int NativeCore::CreateRayTracingProgram(double_ptr<RayTracingProgram>) {
  NativeUnsupported();
}

#if defined(LONGMARCH_CUDA_RUNTIME)
int NativeCore::CreateCUDABuffer(size_t, double_ptr<CUDABuffer>) {
  NativeUnsupported();
}

void NativeCore::CUDABeginExecutionBarrier(cudaStream_t) {
  NativeUnsupported();
}

void NativeCore::CUDAEndExecutionBarrier(cudaStream_t) {
  NativeUnsupported();
}
#endif

}  // namespace grassland::graphics::backend
