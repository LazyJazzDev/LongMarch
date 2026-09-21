#include "sparkium/backend/cuda/cuda_device.h"

#include <iostream>

#include "sparkium/backend/cuda/cuda_util.h"
#include "sparkium/backend/cuda/driver_util.h"
#ifdef SPARKIUM_OPTIX_ENABLED
#include "sparkium/backend/cuda/optix_acceleration_structure.h"
#endif
namespace sparkium::backend {
using namespace cuda;

CudaDevice::~CudaDevice() {
#ifdef SPARKIUM_OPTIX_ENABLED
  optix_.reset();
#endif
#ifdef SPARKIUM_CUDA_ENABLED
  if (cuda_context_) {
    cuCtxSynchronize();
    cuDevicePrimaryCtxRelease(device_index_);
  }
#endif
}

int CudaDevice::GetPhysicalDeviceProperties(PhysicalDeviceProperties *p) {
#ifdef SPARKIUM_CUDA_ENABLED
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
#ifdef SPARKIUM_OPTIX_ENABLED
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

int CudaDevice::InitializeLogicalDevice(int index) {
  int count = GetPhysicalDeviceProperties();
  if (index < 0 || index >= count)
    return -1;
  std::vector<PhysicalDeviceProperties> properties(count);
  GetPhysicalDeviceProperties(properties.data());
  device_name_ = properties[index].name;
  device_index_ = index;
#ifdef SPARKIUM_CUDA_ENABLED
  {
    CUcontext context;
    CheckCUDA(cuDevicePrimaryCtxRetain(&context, index));
    cuda_context_ = context;
    CheckCUDA(cuCtxSetCurrent(context));
#ifdef SPARKIUM_OPTIX_ENABLED
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

void CudaDevice::WaitGPU() {
#ifdef SPARKIUM_CUDA_ENABLED
  CheckCUDA(cuCtxSynchronize());
#endif
}

int CudaDevice::CreateBottomLevelAccelerationStructure(BufferRange,
                                                       uint32_t,
                                                       uint32_t,
                                                       RayTracingGeometryFlag,
                                                       double_ptr<AccelerationStructure>) {
  CudaUnsupported();
}

int CudaDevice::CreateBottomLevelAccelerationStructure(BufferRange vertices,
                                                       BufferRange indices,
                                                       uint32_t vertex_count,
                                                       uint32_t stride,
                                                       uint32_t primitive_count,
                                                       RayTracingGeometryFlag flags,
                                                       double_ptr<AccelerationStructure> output) {
#ifdef SPARKIUM_OPTIX_ENABLED
  if (optix_) {
    output.construct<OptixAccelerationStructure>(optix_.get(), vertices, indices, vertex_count, stride, primitive_count,
                                                 flags);
    return 0;
  }
#endif
  CudaUnsupported();
}

int CudaDevice::CreateBottomLevelAccelerationStructure(Buffer *vertices,
                                                       Buffer *indices,
                                                       uint32_t stride,
                                                       double_ptr<AccelerationStructure> output) {
  if (!vertices || !indices || !stride || vertices->Size() / stride > UINT32_MAX || indices->Size() / 12 > UINT32_MAX)
    throw std::invalid_argument("invalid compute triangle geometry");
  return CreateBottomLevelAccelerationStructure(vertices->Range(), indices->Range(), vertices->Size() / stride, stride,
                                                indices->Size() / 12, RAYTRACING_GEOMETRY_FLAG_NONE, output);
}

int CudaDevice::CreateTopLevelAccelerationStructure(const std::vector<RayTracingInstance> &instances,
                                                    double_ptr<AccelerationStructure> output) {
#ifdef SPARKIUM_OPTIX_ENABLED
  if (optix_) {
    output.construct<OptixAccelerationStructure>(optix_.get(), instances);
    return 0;
  }
#endif
  CudaUnsupported();
}

}  // namespace sparkium::backend
