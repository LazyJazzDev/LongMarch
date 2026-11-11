#pragma once
#include "grassland/rhi/backend/backend_window_state.h"
#include "grassland/rhi/backend/d3d12/d3d12_window.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12WindowImpl final : public Object<ID3D12Window> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::IWindow, ID3D12Window, Window, rhi::IWindow)
  ~D3D12WindowImpl() override;
  static ID3D12Window *Create(ID3D12Device *device, const WindowDesc &desc);
  ObjectPtr<Image> AcquireNativeFrame() override;
  bool RebuildNativeSwapChain(PresentMode mode) override;
  PresentMode GetPresentMode() const override;
  std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const override;
  bool IsFrameValid(uint64_t generation) const;
  HRESULT PresentFrame(uint64_t generation);

 private:
  D3D12WindowImpl(ID3D12Device *device);
  ID3D12Device *device_;
  ComPtr<IDXGISwapChain3> swapchain_;
  std::array<ComPtr<ID3D12Resource>, 3> images_;
  std::array<ObjectPtr<D3D12DescriptorPool::Lease>, 3> attachments_;
  ImageDesc desc_;
  std::atomic<uint64_t> generation_{1};
  bool acquired_{};
  bool valid_{};
  PresentMode mode_{PRESENT_MODE_VSYNC};

 private:
  WindowState state_Window_;

  WindowState &StateWindow() override {
    return state_Window_;
  }

  const WindowState &StateWindow() const override {
    return state_Window_;
  }
};
}  // namespace grassland::rhi::backend::d3d12
