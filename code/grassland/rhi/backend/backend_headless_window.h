#pragma once
#include <condition_variable>
#include <memory>
#include <mutex>

#include "grassland/rhi/backend/backend_window.h"

namespace grassland::rhi::backend {

// Shared with frame leases so releasing a lease never dereferences its window.
class HeadlessFramePool {
 public:
  class Reservation {
   public:
    explicit Reservation(std::shared_ptr<HeadlessFramePool> pool);
    ~Reservation();
    Reservation(const Reservation &) = delete;
    Reservation &operator=(const Reservation &) = delete;

   private:
    std::shared_ptr<HeadlessFramePool> pool_;
  };

  size_t InUse() const;

 private:
  mutable std::mutex mutex_;
  std::condition_variable available_;
  size_t in_use_{};
};

class HeadlessWindow final : public Window {
 public:
  static HeadlessWindow *Create(Device *device, BackendAPI api, const WindowDesc &desc);
  IImage *GetNextImage() override;
  // Drops the window's current-frame reference after CmdPresent retained it.
  void AdvanceFrame();
  bool EnableImGui(bool enabled, const char *font_file_path, float font_size) override;
  bool UpdateImGui(const std::function<void()> &callback) override;
  GLFWwindow *GLFWWindow() const override;
  bool ShouldClose() const override;
  bool IsKeyDown(int key) const override;
  bool SetTitle(const std::string &title) override;
  bool IsSwapChainInvalid() const override;
  bool RebuildSwapChain() override;
  bool UpdateNextImage() override;
  bool SetPresentMode(PresentMode mode) override;
  PresentMode GetPresentMode() const override;
  bool SetColorSpace(ColorSpace color_space) override;
  ColorSpace GetColorSpace() const override;
  bool GetHDRReferenceWhiteNits(float &nits) const override;
  float GetHDRReferenceWhiteScale() const override;
  float GetHDRHeadroom() const override;
  std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const override;
  ObjectPtr<Image> AcquireNativeFrame() override;
  bool RebuildNativeSwapChain(PresentMode mode) override;

 private:
  HeadlessWindow(Device *device, BackendAPI api, const WindowDesc &desc);
  ObjectPtr<Image> CreateFrameLease();
  Device *device_;
  BackendAPI api_;
  uint32_t width_, height_;
  ColorSpace color_space_;
  std::shared_ptr<HeadlessFramePool> pool_;
  ObjectPtr<Image> frame_image_;
  ObjectPtr<Image> current_;
};
}  // namespace grassland::rhi::backend
