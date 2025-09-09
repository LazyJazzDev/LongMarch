#include "cao_di/d3d12/adapter.h"

#include <utility>

namespace CD::d3d12 {

Adapter::Adapter(ComPtr<IDXGIAdapter1> adapter) : adapter_(std::move(adapter)) {
  adapter_->GetDesc1(&desc_);
}

std::string Adapter::Name() const {
  return WStringToString(desc_.Description);
}

SIZE_T Adapter::DedicatedMemorySize() const {
  return desc_.DedicatedVideoMemory;
}

SIZE_T Adapter::DedicatedSystemMemorySize() const {
  return desc_.DedicatedSystemMemory;
}

SIZE_T Adapter::SharedMemorySize() const {
  return desc_.SharedSystemMemory;
}

bool Adapter::SupportRayTracing() const {
  ComPtr<ID3D12Device5> temporal_device;
  if (FAILED(D3D12CreateDevice(adapter_.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&temporal_device)))) {
    return false;
  }
  D3D12_FEATURE_DATA_D3D12_OPTIONS5 feature_support{};
  temporal_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &feature_support, sizeof(feature_support));
  // Check if the adapter supports ray tracing tier 1.1
  return feature_support.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1;
}

uint32_t Adapter::VendorID() const {
  return desc_.VendorId;
}

uint64_t Adapter::Evaluate() const {
  uint64_t score = 0;
  if (desc_.DedicatedVideoMemory > 0) {
    score += desc_.DedicatedVideoMemory / 1024 / 1024;
  }
  if (SupportRayTracing()) {
    score += 100000;
  }
  return score;
}

bool Adapter::CheckFeatureSupport(const DeviceFeatureRequirement &feature_requirement) const {
  if (feature_requirement.enable_raytracing_extension) {
    if (!SupportRayTracing()) {
      return false;
    }
  }
  return true;
}

#if defined(LONGMARCH_CUDA_RUNTIME)
int Adapter::CUDADeviceIndex() const {
  DXGI_ADAPTER_DESC1 adapter_desc = {};
  adapter_->GetDesc1(&adapter_desc);

  int cuda_device_count = 0;
  cudaGetDeviceCount(&cuda_device_count);
  for (int device_index = 0; device_index < cuda_device_count; device_index++) {
    cudaDeviceProp device_prop{};
    cudaGetDeviceProperties(&device_prop, device_index);
    if (std::memcmp(&device_prop.luid, &adapter_desc.AdapterLuid, sizeof(LUID)) == 0) {
      return device_index;
    }
  }
  return -1;
}
#endif

}  // namespace CD::d3d12
