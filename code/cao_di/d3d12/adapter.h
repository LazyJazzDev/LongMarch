#pragma once
#include "cao_di/d3d12/device_feature_requirement.h"
#include "cao_di/d3d12/dxgi_factory.h"

namespace CD::d3d12 {
class Adapter {
 public:
  Adapter(ComPtr<IDXGIAdapter1> adapter);

  IDXGIAdapter1 *Handle() const {
    return adapter_.Get();
  }

  [[nodiscard]] std::string Name() const;
  SIZE_T DedicatedMemorySize() const;
  SIZE_T DedicatedSystemMemorySize() const;
  SIZE_T SharedMemorySize() const;
  bool SupportRayTracing() const;
  uint32_t VendorID() const;
  uint64_t Evaluate() const;

  bool CheckFeatureSupport(const DeviceFeatureRequirement &feature_requirement) const;

#if defined(CHANGZHENG_CUDA_RUNTIME)
  int CUDADeviceIndex() const;
#endif

 private:
  ComPtr<IDXGIAdapter1> adapter_;
  DXGI_ADAPTER_DESC1 desc_{};
};
}  // namespace CD::d3d12
