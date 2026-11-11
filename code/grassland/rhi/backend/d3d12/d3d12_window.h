#pragma once
#include <array>

#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12Window)

namespace grassland::rhi::backend::d3d12 {

class ID3D12Window : public IBackendWindow {
 public:
  static rhi::IWindow *Create(ID3D12Device *device, const WindowDesc &desc);

  virtual bool IsFrameValid(uint64_t generation) const = 0;
  virtual HRESULT PresentFrame(uint64_t generation) = 0;
};

}  // namespace grassland::rhi::backend::d3d12
