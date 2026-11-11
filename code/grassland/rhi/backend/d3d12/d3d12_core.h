#pragma once
#include <d3d12sdklayers.h>

#include "grassland/rhi/backend/backend_core.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12Core)

namespace grassland::rhi::backend::d3d12 {

class ID3D12Core : public IBackendCore {
 public:
  static ID3D12Core *CreateD3D12Core(const CoreDesc &desc);
  virtual std::vector<PhysicalDevice> GetPhysicalDevices() const = 0;
  virtual rhi::IDevice *CreateDevice(int32_t device_idx) = 0;
};

}  // namespace grassland::rhi::backend::d3d12
