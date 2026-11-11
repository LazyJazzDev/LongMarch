#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12Window;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::d3d12::ID3D12Window)
#include <array>

#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {

class ID3D12Window : public Window {
 public:
  static ID3D12Window *Create(ID3D12Device *device, const WindowDesc &desc);
  virtual ObjectPtr<Image> AcquireNativeFrame() = 0;
  virtual bool RebuildNativeSwapChain(PresentMode mode) = 0;
  virtual PresentMode GetPresentMode() const = 0;
  virtual std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const = 0;
  virtual bool IsFrameValid(uint64_t generation) const = 0;
  virtual HRESULT PresentFrame(uint64_t generation) = 0;
};

;

}  // namespace grassland::rhi::backend::d3d12
