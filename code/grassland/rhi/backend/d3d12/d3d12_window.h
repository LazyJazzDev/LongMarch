#pragma once
#include <array>

#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {

class D3D12Window final : public IWindow {
 public:
  explicit D3D12Window(D3D12Device *device);
  ~D3D12Window() override;
  bool Initialize(const WindowDesc &desc);
  ObjectPtr<IImage> AcquireFrame() override;
  bool SetPresentMode(PresentMode mode) override;
  PresentMode GetPresentMode() const override;
  bool IsFrameValid(uint64_t generation) const;
  const void *DeviceIdentity() const;
  void CancelFrame(uint64_t generation);
  HRESULT PresentFrame(uint64_t generation);

 private:
  bool Rebuild(int width, int height);
  D3D12Device *device_;
  ComPtr<IDXGISwapChain3> swapchain_;
  ComPtr<ID3D12DescriptorHeap> rtv_;
  std::array<ComPtr<ID3D12Resource>, 3> images_;
  ImageDesc desc_;
  UINT stride_{};
  std::atomic<uint64_t> generation_{1};
  bool acquired_{};
  bool valid_{};
  PresentMode mode_{PresentMode::VSync};
};

}  // namespace grassland::rhi::backend::d3d12
