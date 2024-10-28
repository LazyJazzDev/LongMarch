#pragma once
#include "grassland/d3d12/adapter.h"

namespace grassland::d3d12 {
class Device {
 public:
  Device(const Adapter &adapter,
         D3D_FEATURE_LEVEL feature_level,
         ID3D12Device *device);

  ID3D12Device *Handle() const {
    return device_.Get();
  }

  Adapter &Adapter() {
    return adapter_;
  }

  D3D_FEATURE_LEVEL FeatureLevel() const {
    return feature_level_;
  }

 private:
  class Adapter adapter_;
  ComPtr<ID3D12Device> device_;
  D3D_FEATURE_LEVEL feature_level_;
};
}  // namespace grassland::d3d12
