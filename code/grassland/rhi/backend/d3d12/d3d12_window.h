#pragma once
#include <array>

#include "grassland/rhi/backend/backend_window.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {

class D3D12Window final : public Window {
 public:
  void AddRef() noexcept override {
    RetainReference();
  }

  void Release() noexcept override {
    if (ReleaseReference())
      delete this;
  }

  bool EnableImGui(bool enabled = true, const char *font_file_path = nullptr, float font_size = 13.0f) override {
    return ConfigureImGui(enabled, font_file_path, font_size);
  }

  bool UpdateImGui(const std::function<void()> &callback) override {
    return RunImGuiFrame(callback);
  }

  EventManager<void(int, int)> &ResizeEvent() override {
    return ResizeCallbacks();
  }

  EventManager<void(int, int)> &FramebufferResizeEvent() override {
    return FramebufferResizeCallbacks();
  }

  EventManager<void(bool)> &CursorEnterEvent() override {
    return CursorEnterCallbacks();
  }

  EventManager<void(bool)> &FocusEvent() override {
    return FocusCallbacks();
  }

  EventManager<void(double, double)> &MouseMoveEvent() override {
    return MouseMoveCallbacks();
  }

  EventManager<void(int, int, int, double, double)> &MouseButtonEvent() override {
    return MouseButtonCallbacks();
  }

  EventManager<void(double, double)> &ScrollEvent() override {
    return ScrollCallbacks();
  }

  EventManager<void(int, int, int, int)> &KeyEvent() override {
    return KeyCallbacks();
  }

  EventManager<void(uint32_t)> &CharEvent() override {
    return CharCallbacks();
  }

  EventManager<void(int, const char **)> &DropEvent() override {
    return DropCallbacks();
  }

  rhi::IImage *GetNextImage() override {
    return PreparedImage();
  }

  bool IsSwapChainInvalid() const override {
    return SwapChainNeedsRebuild();
  }

  bool RebuildSwapChain() override {
    return RecreateSwapChain();
  }

  bool SetPresentMode(PresentMode mode) override {
    return ChangePresentMode(mode);
  }

  bool SetColorSpace(ColorSpace color_space) override {
    return ChangeColorSpace(color_space);
  }

  ColorSpace GetColorSpace() const override {
    return ActiveColorSpace();
  }

  bool GetHDRReferenceWhiteNits(float &nits) const override {
    return QueryHDRReferenceWhiteNits(nits);
  }

  float GetHDRReferenceWhiteScale() const override {
    return HDRReferenceWhiteScale();
  }

  float GetHDRHeadroom() const override {
    return HDRHeadroom();
  }

  GLFWwindow *GLFWWindow() const override {
    return NativeWindow();
  }

  bool ShouldClose() const override {
    return WindowShouldClose();
  }

  bool IsKeyDown(int key) const override {
    return KeyPressed(key);
  }

  bool SetTitle(const std::string &title) override {
    return ChangeTitle(title);
  }

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
  std::array<ObjectPtr<D3D12DescriptorPool::Lease>, 3> attachments_;
  ImageDesc desc_;
  std::atomic<uint64_t> generation_{1};
  bool acquired_{};
  bool valid_{};
  PresentMode mode_{PRESENT_MODE_VSYNC};
};

}  // namespace grassland::rhi::backend::d3d12
