#pragma once
#include <array>

#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {

class D3D12Window final : public Window {
 public:
  ~D3D12Window() override;
  static D3D12Window *Create(D3D12Device *device, const WindowDesc &desc);
  ObjectPtr<Image> AcquireNativeFrame() override;
  bool RebuildNativeSwapChain(PresentMode mode) override;
  PresentMode GetPresentMode() const override;
  std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const override;
  bool IsFrameValid(uint64_t generation) const;
  const void *DeviceIdentity() const;
  HRESULT PresentFrame(uint64_t generation);

 private:
  D3D12Window(D3D12Device *device);

  D3D12Device *device_;
  ComPtr<IDXGISwapChain3> swapchain_;
  std::array<ComPtr<ID3D12Resource>, 3> images_;
  ImageDesc desc_;
  std::atomic<uint64_t> generation_{1};
  bool acquired_{};
  bool valid_{};
  PresentMode mode_{PRESENT_MODE_VSYNC};
};

}  // namespace grassland::rhi::backend::d3d12
