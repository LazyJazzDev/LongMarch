#pragma once
#include "grassland/d3d12/d3d12util.h"
#include "grassland/util/double_ptr.h"

namespace grassland::d3d12 {
class DXGIFactory {
 public:
  explicit DXGIFactory(IDXGIFactory4 *factory);

  IDXGIFactory4 *Handle() const {
    return factory_.Get();
  }

  std::vector<Adapter> EnumerateAdapters() const;

  HRESULT CreateDevice(
      const DeviceFeatureRequirement &device_feature_requirement,
      int device_index,
      double_ptr<Device> pp_device);

 private:
  ComPtr<IDXGIFactory4> factory_;
};

HRESULT CreateDXGIFactory(double_ptr<DXGIFactory> pp_factory);

}  // namespace grassland::d3d12
