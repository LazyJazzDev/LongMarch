#pragma once
#include "grassland/graphics/backend/d3d12/helper/d3d12util.h"
#include "grassland/util/double_ptr.h"

namespace grassland::graphics::backend::d3d12 {

struct DXGIFactoryCreateHint {
  bool enable_debug{kDefaultEnableDebugLayer};
  DXGIFactoryCreateHint(bool enable_debug = kDefaultEnableDebugLayer);
};

class DXGIFactory {
 public:
  explicit DXGIFactory(DXGIFactoryCreateHint hint, const ComPtr<IDXGIFactory4> &factory);

  IDXGIFactory4 *Handle() const {
    return factory_.Get();
  }

  std::vector<Adapter> EnumerateAdapters() const;

  HRESULT CreateDevice(const DeviceFeatureRequirement &device_feature_requirement,
                       int device_index,
                       double_ptr<Device> pp_device);

 private:
  DXGIFactoryCreateHint hint_;
  ComPtr<IDXGIFactory4> factory_;
};

HRESULT CreateDXGIFactory(DXGIFactoryCreateHint hint, double_ptr<DXGIFactory> pp_factory);

}  // namespace grassland::graphics::backend::d3d12
